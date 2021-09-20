import time
import torch
import torchvision
import itertools
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import scipy
import cv2
import kornia as K
import sys
import kornia.geometry as G
import imgaug

from torchvision import transforms
from scipy import stats
from scipy.stats import norm, rv_continuous
from tqdm import tqdm, tqdm_notebook
from IPython import display
from kornia.filters.gaussian import GaussianBlur2d

torch.set_printoptions(precision=10)


# +
def random_augment(batch, type_of_augmentation, 
            params_of_augmentation = {
                'degrees' : 45.0,
                'translation' : (0.25, 0.25),
                'gamma': 0.5,
                'resample': 'bilinear',
                'sigma': 1.0,
                'awgn_normed': False,
                'brightness': 0.0,
                'scale': (1.0, 1.0),
                'contrast': 1.0,
                'blur_kernel': (3, 3),
                'blur_sigma': (1.0, 1.0)
            }, do_clip=True):
    '''
    batch is a set of imgs scaled to [0,1]
    type_of_augmentation is a string coding a sequence of augments,
    i.e. 'rotation-rotation' stays for augmentation with rotation applied twice
    
    params_of_augmentation is a dict with parameters of augmentation procedure
    '''
    seq_of_augmentations = [aug for aug in type_of_augmentation.split('-')]
    int_result = batch#.clone()
    device = int_result.device
    for random_augmentation in seq_of_augmentations:
        if random_augmentation == 'blur':
#             s_min = 1e-10 #sigma may only be poitive, so for sampling the small mimimum value is hardcoded
#             s = (s_min, s_min)
#             params_of_augmentation['blur_sigma'] = tuple(np.maximum(s, params_of_augmentation['blur_sigma']))
            
#             s1_max, s2_max = params_of_augmentation['blur_sigma']
#             s1, s2 = (s_min - s1_max) * torch.rand(1).item() + s1_max, (s_min - s2_max) * torch.rand(1).item() + s2_max
#             s = (s1, s2)
#             kernel_size = tuple(map(int, np.ceil(3*np.array(s)) // 2 * 2 + 1)) #like in OpenCV
            blurer = imgaug.augmenters.blur.GaussianBlur(sigma = (0.0, params_of_augmentation['blur_sigma'][0]))
            int_result = blurer.augment(images=int_result.cpu().numpy().transpose(0,2,3,1))
            int_result = torch.from_numpy(int_result.transpose(0,3,1,2)).to(device)
        
        if random_augmentation == 'rotation':
            int_result = K.augmentation.RandomRotation(degrees=params_of_augmentation['degrees'], p=1.0)(int_result)

        if random_augmentation == 'translation':
            eps_degrees = 1e-10
            int_result = K.augmentation.RandomAffine(degrees=(0., eps_degrees), translate=params_of_augmentation['translation'], p=1.0)(int_result)
#             v1, v2 = params_of_augmentation['translation']
#             t = (-2 * v1 * torch.rand(1).item() + v1, -2 * v2 * torch.rand(1).item() + v2)
#             translation_t = (torch.tensor(t, device=device)*torch.tensor(int_result.shape[-2:], device=device)).expand(len(int_result), -1)
#             int_result = G.translate(int_result, translation_t)

        if random_augmentation == 'gamma':
            min_g, max_g = min(params_of_augmentation['gamma'], 1/params_of_augmentation['gamma']), max(params_of_augmentation['gamma'], 1/params_of_augmentation['gamma'])
            g = (min_g - max_g) * torch.rand(len(int_result)) + max_g
            int_result = K.enhance.adjust.adjust_gamma(int_result, g)
        
        if random_augmentation == 'awgn':
            noise = torch.randn_like(int_result)# * params_of_augmentation['sigma']
            if params_of_augmentation['awgn_normed'] is True:
                noise = noise / torch.norm(noise, p=2)
            int_result = int_result + noise * params_of_augmentation['sigma']
            
        if random_augmentation == 'brightness':
            b = -2*params_of_augmentation['brightness'] * torch.rand(len(int_result)) + params_of_augmentation['brightness']
            int_result = K.enhance.adjust.adjust_brightness(int_result, b)
            int_result = int_result.clip(min=0.0, max=1.0)
            
        if random_augmentation == 'scale': 
            v1, v2 = params_of_augmentation['scale']
            d1, d2 = np.abs(1.0 - v1), np.abs(1.0 - v2)
            min_1, max_1 = 1.0-d1, 1.0+d1
            min_2, max_2 = 1.0-d2, 1.0+d2
            s = ((min_1 - max_1) * torch.rand(len(int_result)) + max_1, (min_2 - max_2) * torch.rand(len(int_result)) + max_2)
            scale_s = (torch.stack(s).T).to(device)#torch.tensor(s, device=device).expand(len(int_result), -1)
            int_result = K.scale(int_result, scale_s)
            
        if random_augmentation == 'contrast':
            c = params_of_augmentation['contrast']
            d = np.abs(1.0 - c)
            min_c, max_c = 1.0 -d , 1.0+d
            c = (min_c - max_c) * torch.rand(len(int_result)) + max_c
            int_result = K.enhance.adjust.adjust_contrast(int_result, c)
            int_result = int_result.clip(min=0.0, max=1.0)
            
    if do_clip is True:
        return int_result.clip(min=0.0, max=1.0)
    else:
        return int_result


# -

class synthetic_transform:
    def __init__(self, 
                 arg_params,
                type_of_transform = [],
                 params = []
                ):
        #type_of_transform - list of str types splitted by ' ', e.g. ['rotation', 'translation', 'gamma'] 
        #params is list of parameters for corresponding deterministic transform, e.g. [10.0, (2,2), 0.5]
            self.type = type_of_transform
            self.params = params
            self.arg_params = arg_params
    def act(self, tensor, params=None):        
        if params is None:
            params = self.params
        '''
        tensor to be of shape BxCxHxW
        '''
        device = tensor.device
        tensor_t = tensor#.clone()
        for transform, theta in zip(self.type, params):
            if transform == 'blur':
                s_min = (1e-10, 1e-10) #min small value
                theta_blur = tuple(np.maximum(s_min, theta))

                blurer = imgaug.augmenters.blur.GaussianBlur(sigma = (max(theta_blur)))
                tensor_t = blurer.augment(images=tensor_t.cpu().numpy().transpose(0,2,3,1))
                tensor_t = torch.from_numpy(tensor_t.transpose(0,3,1,2)).to(device)
            if transform == 'rotation':
                theta = torch.tensor(theta).to(device)
                tensor_t = G.rotate(tensor_t, theta)
            if transform == 'translation':
                translation_theta = (((torch.tensor(theta, device=device))*torch.tensor(tensor_t.shape[-2:], device=device)).expand(len(tensor_t), -1)).float()
                
                tensor_t = G.translate(tensor_t, translation_theta)
            if transform == 'gamma':
                tensor_t = K.enhance.adjust.adjust_gamma(tensor_t, theta)
            if transform == 'contrast':
                tensor_t = K.enhance.adjust.adjust_contrast(tensor_t, theta)
                tensor_t = tensor_t.clip(min=0.0, max=1.0)
            if transform == 'brightness':
                tensor_t = K.enhance.adjust.adjust_brightness(tensor_t, theta)
                tensor_t = tensor_t.clip(min=0.0, max=1.0)
            if transform == 'scale':
                scale_theta = torch.tensor(theta, device=device).expand(len(tensor_t), -1)
                tensor_t = K.scale(tensor_t, scale_theta)
        return tensor_t


def inference(model, inf_type, sample, num_of_transforms, type_of_transform, params={
                'degrees' : 45.0,
                'translation' : (0.25, 0.25),
                'gamma': 0.5,
                'resample': 'bilinear',
                'sigma': 1.0,
                'awgn_normed': False,
                'brightness': 0.0,
                'scale': (1.0, 1.0),
                'contrast': 1.0
            }):
    model.eval()
    
    if inf_type == 'plain':
        if len(sample.shape) < 4:
            sample = sample.expand(1, *sample.shape) #add batchsize dim
        logits = model(sample)
        logits = logits[0]
        probs = torch.nn.Softmax(dim=-1)(logits)
        return probs
    
    if inf_type == 'smoothing':
        batch = sample.expand(num_of_transforms, *sample.shape)
        batch = random_augment(batch, type_of_augmentation=type_of_transform, params_of_augmentation=params)
        logits = model(batch)
        logits = torch.mean(logits, dim=0)
        probs = torch.nn.Softmax(dim=-1)(logits)
        return probs
    
    if inf_type == 'aug^2':
        batch = sample.expand(num_of_transforms, *sample.shape)
        batch = random_augment(batch, type_of_augmentation=type_of_transform+' '+type_of_transform, params_of_augmentation=params
                              )
        logits = model(batch)
        logits = torch.mean(logits, dim=0)
        probs = torch.nn.Softmax(dim=-1)(logits)
        return probs
    return None            



def observations_for_chernoff_bound(model, inf_type, sample, y_true, 
                    num_of_sample_transforms, num_of_samples, type_of_transform, params, 
                                    temperature=None, 
                                    default_params={
                                                    'degrees' : 45.0,
                                                    'translation' : (0.25, 0.25),
                                                    'gamma': 0.5,
                                                    'resample': 'bilinear',
                                                    'sigma': 1.0,
                                                    'awgn_normed': False
                                                }, maxphi=10.0):
    '''
    computes right side of Bernstein's bound: Pr(Y>=a) <= E(e^tY) / e^ta
    Y = Pr_label(x) - max_{i!=label}Pr_i(x)
    init bound: a = [Pr_label(sample) - max_{i!=label}Pr_i(sample)] / 2
    
    '''
    y_true = y_true.item()
    Y = torch.zeros(num_of_samples)
    
    probs = inference(model, inf_type, sample, num_of_sample_transforms, type_of_transform, default_params)
    p_1 = probs[y_true] #true probability
    probs_without_y_true = probs.clone()
    probs_without_y_true[y_true] -= 3*probs_without_y_true[y_true] #to make it non-maximum
    probs_without_y_true = probs_without_y_true.sort(dim=-1, descending=True)
    p_2 = probs_without_y_true[0][0] #closest wrong prob
    a = (p_1-p_2)/2.0
    Y_0 = torch.tensor([p_1, p_2])
    space = np.linspace(-maxphi, maxphi, num_of_samples)
    for i in range(num_of_samples):
        new_sample = sample.expand(1, *sample.shape)
        new_sample = random_augment(new_sample, type_of_augmentation=type_of_transform).squeeze(0)
#         transformer = synthetic_transform(type_of_transform='rotation')
#         new_sample = transformer.act(new_sample, params = {
#                     'degrees': torch.tensor([space[i], space[i]]),
#                     'translations': torch.tensor([0.0, 0.0]),
#                     'gamma': 1.0,
#                  }).squeeze(0)
        
        probs = inference(model, inf_type, new_sample, num_of_sample_transforms, type_of_transform, params)
        p_1 = probs[y_true] #true probability
        probs_without_y_true = probs.clone()
        probs_without_y_true[y_true] -= 3*probs_without_y_true[y_true] #to make it non-maximum
        probs_without_y_true = probs_without_y_true.sort(dim=-1, descending=True)
        p_2 = probs_without_y_true[0][0] #closest wrong prob
        Y_i = torch.tensor([p_1, p_2])
        Y[i] = torch.norm(Y_0 - Y_i, p=float('inf'))
    
    #E_exp_ty = torch.mean(torch.exp(temperature*Y))
    #exp_ta = torch.exp(temperature*a)
    #bound = E_exp_ty / exp_ta
    
    return Y, a

def chernoff_bound_on_samples(samples, t, a):
    E_exp_ty = torch.mean(torch.exp(t*samples))
    exp_ta = torch.exp(t*a)
    return E_exp_ty / exp_ta


def cohen_certified_radius(sigma_of_smoothers, p1, p2):
    return sigma_of_smoothers/2.0 * (norm.ppf(p1) - norm.ppf(p2))


def empirical_accuracy(model, loader, transform, support_parameters_dict, device):
    with torch.no_grad():
        transforms = transform.split('-')
        _transformer = synthetic_transform(arg_params = support_parameters_dict, type_of_transform=transforms)
        emp_acc = 0
        parameters = get_parameters(transforms, support_parameters_dict)
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)
            ok_on_transforms = torch.tensor([True]*len(images)).to(device)
            for p in parameters:
                #print(p)
                img_T = _transformer.act(images, p)
                #print((img_T - images).norm())
                y_pred = torch.nn.Softmax(dim=-1)(model(img_T)).argmax(dim=-1)
                ok_on_transforms = ((y_pred == labels) * ok_on_transforms)
            emp_acc += sum(ok_on_transforms)
        return emp_acc / len(loader.dataset)


def get_parameters(transform: list, support_parameters_dict: dict,
                   grid_dicretesation = {'rotation': 20,
                                         'translation': 20,
                                         'brightness': 20,
                                         'contrast': 20,
                                         'scale': 20,
                                         'gamma': 20,
                                         'blur': 20
                   }):
    global_params = []
    for t in transform:
        if t == 'blur':
            min_blur_sigma = (1e-10, 1e-10)
            max_blur_sigma = tuple(np.maximum(min_blur_sigma, support_parameters_dict['blur_sigma']))
            b1, b2 = np.linspace(min_blur_sigma[0], max_blur_sigma[0], grid_dicretesation[t]), np.linspace(min_blur_sigma[1], max_blur_sigma[1], grid_dicretesation[t])
            b = list(itertools.product(b1,b2,repeat=1))
            global_params.append(b)
        if t == 'rotation':
            phis = np.linspace(-support_parameters_dict['degrees'], support_parameters_dict['degrees'], grid_dicretesation[t])
            global_params.append(phis)
        if t == 'translation':
            t1 = np.linspace(-support_parameters_dict[t][0], support_parameters_dict[t][0], grid_dicretesation[t])
            t2 = np.linspace(-support_parameters_dict[t][1], support_parameters_dict[t][1], grid_dicretesation[t])
            t = list(itertools.product(t1,t2, repeat=1))
            global_params.append(t)
        if t == 'brightness':
            b = np.linspace(-support_parameters_dict[t], support_parameters_dict[t], grid_dicretesation[t])
            global_params.append(b)
        if t == 'contrast':
            c = support_parameters_dict[t]
            d = np.abs(1.0 - c)
            min_c, max_c = 1.0 -d , 1.0+d
            ts = np.linspace(min_c, max_c, grid_dicretesation[t])
            global_params.append(ts)
        if t == 'scale':
            v1, v2 = support_parameters_dict[t]
            d1, d2 = np.abs(1.0 - v1), np.abs(1.0 - v2)
            min_1, max_1 = 1.0-d1, 1.0+d1
            min_2, max_2 = 1.0-d2, 1.0+d2
            s1 = np.linspace(min_1, max_1, grid_dicretesation[t])
            s2 = np.linspace(min_2, max_2, grid_dicretesation[t])
            s = list(itertools.product(s1, s1, repeat=1))
            global_params.append(s)
        if t == 'gamma':
            min_c, max_c = min(support_parameters_dict[t], 1/support_parameters_dict[t]), max(support_parameters_dict[t], 1/support_parameters_dict[t])
            ts = np.linspace(min_c, max_c, grid_dicretesation[t])
            global_params.append(ts)
    all_the_parameters = list(map(list, itertools.product(*global_params, repeat=1)))
    return all_the_parameters
