# +
import argparse
import wandb
import os
import random
import setGPU
import json
from code.datasets import get_dataset, get_dataloader, get_num_classes, DATASETS
from time import time
import torch
import torchvision
import datetime

from code.aug_aug import inference, cohen_certified_radius, \
    chernoff_bound_on_samples, observations_for_chernoff_bound, \
    random_augment, synthetic_transform, empirical_accuracy, get_parameters
from code.models import ResNet18Cifar10, get_architecture, ARCHITECTURES
from code.bounds import *
from code.utils import *

# +
#wandb.login()

# +
parser = argparse.ArgumentParser(description='Certify many examples')


parser.add_argument("--model_type", type=str, choices=['plain'], help="type of inference of model")
parser.add_argument("--train_type", type=str, choices=['plain', 'smoothed'], help="type of training of model")
parser.add_argument("--arch_type", type=str, choices=ARCHITECTURES, help="which architecture to pick")
parser.add_argument("--dataset", type=str, choices=DATASETS, help="which dataset")
parser.add_argument("--base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--nsamples", type=int, default=100)
parser.add_argument("--nbounds", type=int, default=10)
parser.add_argument("--transform", type=str, default='rotation')
parser.add_argument("--sigma", type=float, help="noise hyperparameter", default=8./255)
parser.add_argument("--degrees", type=float, help="degree angle \phi, rotation from [-phi, [phi]", default=10.0)
parser.add_argument("--translation", nargs='+', type=float, help="components of maximum translation vector (v1, v2)", default=[1.0, 1.0])
parser.add_argument("--gamma", type=float, help="gamma correction coefficient", default=1.0)
parser.add_argument("--brightness", type=float, help="brightness correction coefficient", default=0.0)
parser.add_argument("--scale", nargs='+', type=float, help="scale correction coefficients", default=[1.0, 1.0])
parser.add_argument("--blur_sigma", nargs='+', type=float, help="tuple of sigmas for gaussian blur", default=[1.0, 1.0])
parser.add_argument("--blur_kernel", nargs='+', type=int, help="tuple of kernel size for gaussian blur", default=[3, 3])
parser.add_argument("--contrast", type=float, help="contrast factor for contrast adjustment transform", default=1.0)
parser.add_argument("--delta", type=int, default=0.9)
parser.add_argument("--num_aug", type=int, default=100)

args = parser.parse_args()

# +
if __name__ == "__main__":
    # load the base classifier
    
    arch_type = args.arch_type
    checkpoint = torch.load(args.base_classifier)
    
    model_type = args.model_type
    num_classes = get_num_classes(args.dataset)
    noise_sd = 0.01
    lr = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_architecture(arch_type, num_classes)
    
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    nsamples = args.nsamples
    nbounds = args.nbounds
    transform = args.transform
    delta = args.delta
    num_aug = args.num_aug
    
    params = {'sigma': args.sigma,
             'awgn_normed': False,
             'degrees': args.degrees,
             'translation': tuple(args.translation),
             'gamma': args.gamma,
             'brightness': args.brightness,
             'scale': tuple(args.scale),
             'blur_kernel': tuple(args.blur_kernel),
             'blur_sigma': tuple(args.blur_sigma),
              'contrast': args.contrast,
             'resample': 'bilinear',
             'nsamples': args.nsamples,
             'nbounds': args.nbounds,
             'transform': args.transform,
             'delta': args.delta,
             'num_aug': args.num_aug,
             'model_type': model_type}
    print(params)
    
#     wandb.init(.....)
#     config = wandb.config
    
    dataloader = get_dataloader(args.dataset, args.split, args.batch)
    
    _bounds, _deltas, _masks, _attacked = get_bounds(model, inf_type=model_type, dataloader=dataloader, num_aug=num_aug,
                                          transform=transform, nbounds=nbounds, delta=delta, nsamples=nsamples,
                                          tmin=1e-4, tmax=1e4, tsteps=500, params=params,
                                          batchwise=True, batch_instead=False, do_clip=True)
    
    bds = []
    list_th = [10**k for k in range(-10, -1, 1)]
    for th in list_th:
        bound_rob_acc = 0
        for b, p in zip(_bounds.view(-1), _masks):
            if b < th and p.item() is True:
                bound_rob_acc += 1
        bound_rob_acc /= len(_bounds)
        bds.append(bound_rob_acc)
     
    e_a = empirical_accuracy(model, dataloader, transform, params, device)
    
#     wandb.log({"bounds":[bd for bd in _bounds], 
#                "deltas":[d for d in _deltas], 
#                "masks":[m for m in _masks],
#               "support_for_bound_acc": list_th,
#               "bound_accuracy": bds,
#               "empirical_accuracy": e_a,
#              "attacked": [a for a in _attacked]
#               })
    

#     # prepare output file
filepath = 'results/' + args.dataset + '/' + args.arch_type + '/' + args.train_type +'/' + args.transform  + '/'+ args.outfile
if not os.path.isdir(filepath):
    os.makedirs(filepath)
torch.save(_bounds, filepath + '/' + 'bounds.pt')
with open(filepath+ '/' + 'parameters.json', 'w') as fp:
    json.dump(params, fp)
torch.save(_masks, filepath + '/' + 'correct_pred.pt')
