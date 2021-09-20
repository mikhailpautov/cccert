import torch
import numpy as np

from tqdm import tqdm

from .aug_aug import random_augment
from .fmt import get_mag

default_params={
    'degrees' : 45.0,
    'translation' : (0.25, 0.25),
    'gamma': 0.5,
    'resample': 'bilinear',
    'sigma': 8/255,
    'awgn_normed': False
}


def inference(model, inf_type, batch, num_aug, aug, params=default_params, do_clip=True):
    model.eval()
    if len(batch.shape) < 4:
        batch = batch.expand(1, *batch.shape) #add batchsize dim
        
    if inf_type == 'plain':
        logits = model(batch)
        logits = logits
        probs = torch.nn.Softmax(dim=-1)(logits)
        return probs

    if inf_type == 'smoothing':
        batch_size = batch.shape[0]
        batch = torch.repeat_interleave(batch, num_aug, 0)
        batch = random_augment(batch, type_of_augmentation=aug,
                               params_of_augmentation=params, do_clip=do_clip)
        logits = model(batch)
        logits = logits.view(batch_size, num_aug, -1)
        logits = torch.mean(logits, dim=1)
        probs = torch.nn.Softmax(dim=-1)(logits)
        return probs

    if inf_type == 'aug^2':
        batch_size = batch.shape[0]
        batch = torch.repeat_interleave(batch, num_aug, 0)
        batch = random_augment(batch, ' '.join([aug,aug]), params, do_clip=do_clip)
        logits = model(batch)
        logits = logits.view(batch_size, num_aug, -1)
        logits = torch.mean(logits, dim=1)
        probs = torch.nn.Softmax(dim=-1)(logits)
        return probs
    return None


def get_bounds(model, inf_type, dataloader, num_aug, transform, nbounds, delta, nsamples=500,
               tmin=1e-4, tmax=50, tsteps=50, params=default_params,
               batchwise=False, batch_instead=False, do_clip=True):
    '''
    Computes min_t bounds for t in interval [tmin, tmax] with tsteps steps.

    Arguments
    model: model to test
    inf_type: inference type ('plain', 'smoothing', 'aug^2')
    dataloader: a dataloader of the test/train set
    num_aug: number of augmentations to perform in inference ('smoothing',
    'aug^2')
    transform: type of transform performed ('rotation', 'translation', 'awgn',
    'gamma')
    nbounds: int, how many bounds per sample to compute
    delta: float, the threshold parameter in Paley-Zygmund inequality, "P(X<delta*E(X)) <= ..."
    nsamples: number of samples for expectation in Bernstein's bound
    tmin: min value for temperature interval 
    tmax: max value for temperature interval
    tsteps: number of steps in the temperature range
    params: a dictionary with parameters of transforms
    batch_instead: bool, to work on batch instead of loader
    do_clip: bool, whether to return clipped img

    Returns
    bounds: minimal over t values bounds for each image
    deltas: top 2 prediction probabilities difference for each image
    '''

    t = torch.linspace(tmin, tmax, tsteps).cuda()
    bounds = []
    deltas = []
    hitmask = []
    attacked = []

    with torch.no_grad():
        if batch_instead is False:
            for batch in tqdm(dataloader):
                Y, a, mask, attacked_on_batch = obs_chernoff(model, inf_type, batch,
                    num_aug, transform, nbounds*nsamples, params, batchwise, do_clip)
                
                Y_ = Y.view(-1, nbounds*nsamples)
                Yt = torch.einsum('ij,k->ijk', Y_, t)
                Yt = Yt.reshape(-1, nbounds, nsamples, tsteps)
                E_exp_ty = torch.mean(torch.exp(Yt), 2)
                exp_ta = torch.exp(torch.outer(a, t))
                exp_ta = exp_ta.unsqueeze(1).repeat(1, nbounds, 1)
                R = E_exp_ty / exp_ta
                min_bounds, temp_idx = torch.topk(R, 1, dim=2, largest=False)
                max_bounds = torch.max(min_bounds, dim=1).values/delta

                bounds.append(max_bounds)
                deltas.append(2*a)
                hitmask.append(mask)
                attacked.append(attacked_on_batch)

        else:
            Y, a, mask, attacked_on_batch = obs_chernoff(model, inf_type, dataloader,
                    num_aug, transform, nbounds*nsamples, params, batchwise, do_clip)
            Y_ = Y.view(-1, nbounds*nsamples)
            Yt = torch.einsum('ij,k->ijk', Y_, t)
            Yt = Yt.reshape(-1, nbounds, nsamples, tsteps)
            E_exp_ty = torch.mean(torch.exp(Yt), 2)
            exp_ta = torch.exp(torch.outer(a, t))
            exp_ta = exp_ta.unsqueeze(1).repeat(1, nbounds, 1)
            R = E_exp_ty / exp_ta
            min_bounds, temp_idx = torch.topk(R, 1, dim=2, largest=False)
            max_bounds = torch.max(min_bounds, dim=1).values/delta
           
            bounds.append(max_bounds)
            deltas.append(2*a)
            hitmask.append(mask)
            attacked.append(attacked_on_batch)
    attacked = torch.hstack(attacked)
    bounds = torch.vstack(bounds)
    deltas = torch.hstack(deltas)
    hitmask = torch.hstack(hitmask)
    
    return torch.min(torch.ones_like(bounds), bounds), deltas, hitmask, attacked


def get_range_bounds(model, inf_type, dataloader, num_aug, transform,
        span_param, span, nsamples=500, tmin=1e-4, tmax=50, tsteps=50,
        params=default_params, batchwise=False):

    '''
    Same as get_bounds, but perorms bound computations over a range of
    transform parameters (e.g. angles: [0,5,10,...])

    Arguments
    span_param: a string denoting over which parameter to perform bound
        computation (e.g. 'degrees', 'sigma')
    span: an array with range values

    Returns
    bounds: a [span.size() x len(dataloader.dataset)]  array of bounds
    deltas: a [span.size() x len(dataloader.dataset)] array of top2 predictions
        differences
    '''

    t = torch.linspace(tmin, tmax, tsteps).cuda()

    bounds = np.zeros((len(dataloader.dataset), len(span)))
    deltas = np.zeros((len(dataloader.dataset), len(span)))
    hitmask = np.zeros((len(dataloader.dataset)))
    counter = 0
    for batch in tqdm(dataloader):
        counter_prev = counter
        counter += batch[0].shape[0]
        with torch.no_grad():
            for i, value in enumerate(span):
                params[span_param] = value
                Y, a, mask = obs_chernoff(model, inf_type, batch,
                        num_aug, transform, nsamples, params, batchwise)

                Y_ = Y.view(-1, nsamples)

                Yt = torch.einsum('ij,k->ijk', Y_, t)
                E_exp_ty = torch.mean(torch.exp(Yt), 1)
                exp_ta = torch.exp(torch.outer(t, a))
                R = E_exp_ty / exp_ta.T
                min_bounds, temp_idx = torch.topk(R, 1, dim=1, largest=False)
                bounds[counter_prev:counter, i] = min_bounds.squeeze().cpu()
                deltas[counter_prev:counter, i] = 2*a.cpu()
                hitmask[counter_prev:counter] = mask.cpu() # same for all i

    return bounds, deltas, hitmask

def obs_chernoff(model, inf_type, batch, num_aug, transform, nsamples, params,
        batchwise=True, do_clip=True, max_samples_for_batch=1000):
    '''
    Computes the right side of bernstein's bound:
    pr(y>=a) <= e(e^ty) / e^ta,
    where y = \| p - p_t \|_{\inf}, where p = f(x) and p_t = f(x_t),
    x_t is a transformed input x,
    f is a network,
    a = [p1 - p2] / 2, p1, p2 = top2(p).
    '''
    model.eval()
    with torch.no_grad():
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()
        batch_size = images.shape[0]
        probs0 = inference(model, inf_type, images, num_aug, transform, params, do_clip)
        pred = probs0.data.max(1)[1]
        hitmask = pred == labels
        y0, idx0 = torch.topk(probs0, 2, dim=1)
        a = (y0[:,0] - y0[:,1])/2

        num_of_tries = nsamples // max_samples_for_batch
        remainder = nsamples % max_samples_for_batch
        
        full_Y = None
        not_attacked = None
        if num_of_tries > 0:
            for _ in range(num_of_tries):
                new_images = torch.repeat_interleave(images, max_samples_for_batch, 0)
                new_images = random_augment(new_images, transform, params, do_clip)
                if batchwise:
                    probs1 = inference(model, inf_type, new_images, num_aug, transform, params, do_clip)

                    probs1_for_cp = probs1.argmax(dim=-1)
                    probs1_for_cp = probs1_for_cp.reshape(batch_size, max_samples_for_batch)

                    probs1 = probs1.reshape(batch_size, max_samples_for_batch, -1)
                    probs0_for_comparison = probs0.unsqueeze(1).repeat(1, max_samples_for_batch, 1)
                    labels_c = labels.unsqueeze(1)
                    labels_c = labels_c.expand(-1, max_samples_for_batch)

                    now_not_attacked = torch.sum(torch.eq(probs1_for_cp, labels_c), dim=-1)
                    if not_attacked is None:
                        not_attacked = now_not_attacked
                    else:
                        not_attacked += now_not_attacked
                    diff = probs0_for_comparison - probs1
                    Y = torch.linalg.norm(diff, float('inf'), dim=-1)
                    if full_Y is None:
                        full_Y = Y
                    else:
                        full_Y = torch.hstack([full_Y, Y])
                else:
                    raise NotImplementedError('use batchwise intead, but add extra dim')
                
        if remainder > 0:        
            new_images = torch.repeat_interleave(images, remainder, 0)
            new_images = random_augment(new_images, transform, params, do_clip)

            if batchwise:
                probs1 = inference(model, inf_type, new_images, num_aug, transform, params, do_clip)
                probs1 = probs1.reshape(batch_size, remainder, -1)
                probs0_for_comparison = probs0.unsqueeze(1).repeat(1, remainder, 1)
                labels_c = labels.unsqueeze(1)
                labels_c = labels_c.expand(-1, remainder) 
                now_not_attacked = torch.sum(torch.eq(probs1.argmax(dim=-1), labels_c), dim=-1)
                if not_attacked is None:
                    not_attacked = now_not_attacked
                else:
                    not_attacked += now_not_attacked
                diff = probs0_for_comparison - probs1
                Y = torch.linalg.norm(diff, float('inf'), dim=-1)
                if full_Y is None:
                    full_Y = Y
                else:
                    full_Y = torch.hstack([full_Y, Y])
            else:
                raise NotImplementedError('use batchwise intead, but add extra dim')
        attacked = nsamples - not_attacked
        
    return full_Y, a, hitmask, attacked
