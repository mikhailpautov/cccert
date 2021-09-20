import math

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F


def to_range(theta, w):
    '''
    range of (-pi/2, pi/2)
    '''
    angle = -(float(theta) * 180.0 ) / w
    if angle < - 90.:
        angle += 180
    else:
        if angle > 90.:
            angle -= 180
    pred_theta = angle / 180 * math.pi

    return angle, pred_theta

def rotmat(theta):
    '''
    Creates rotation matrix for angle theta (in radians).
    '''
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def rotate(x, theta, padding_mode='zeros'):
    '''
    Rotates a batch of images in `x` by angles theta.

    Args:
    x: batch of images in 4D [batch_size, channels, width, height]
    theta: (torch.tensor) angles for rotation in radians.
    '''
    if theta.numel() == 1:
        rot_mat = rotmat(theta)[None, ...].repeat(x.shape[0],1,1).to(x.device)
    else:
        rot_mat = rotmats(theta)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=False)
    return x

def rotmats(theta):
    '''
    Creates rotation matrices for multiple angles in theta.
    '''
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    cs = torch.stack((cos, sin), 1)
    sc = torch.stack((-sin, cos), 1)
    zeros = torch.zeros_like(cs)
    rotmats = torch.squeeze(torch.stack((cs, sc, zeros), 2))
    return rotmats

def rotate_one(x, theta):
    '''
    Creates multiple rotations for one image in x.
    '''
    x = x.repeat(theta.shape[0],1,1,1)
    rot_mats = rotmats(theta)
    grid = F.affine_grid(rot_mats, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, mode='nearest', align_corners=False)
    return x


def unravel_indices(indices, shape) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.
    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim
    coord = torch.stack(coord[::-1], dim=-1)
    return coord

# TODO doesn't work properly for now
def map_coordinates(x, coordinates):
    ''' PyTorch version of scipy.ndimage.interpolation.map_coordinates
    input: (H, W)
    coordinates: (2, ...)
    '''
    h = x.shape[-2]
    w = x.shape[-1]

    def _coordinates_pad_wrap(h, w, coordinates):
        coordinates[0] = coordinates[0] % h
        coordinates[1] = coordinates[1] % w
        return coordinates

    co_floor = torch.floor(coordinates).long()
    co_ceil = torch.ceil(coordinates).long()
    d1 = (coordinates[1] - co_floor[1].float())
    d2 = (coordinates[0] - co_floor[0].float())
    co_floor = _coordinates_pad_wrap(h, w, co_floor)
    co_ceil = _coordinates_pad_wrap(h, w, co_ceil)
    f00 = x[:, co_floor[0], co_floor[1]]
    f10 = x[:, co_floor[0], co_ceil[1]]
    f01 = x[:, co_ceil[0], co_floor[1]]
    f11 = x[:, co_ceil[0], co_ceil[1]]
    fx1 = f00 + d1 * (f10 - f00)
    fx2 = f01 + d1 * (f11 - f01)
    return fx1 + d2 * (fx2 - fx1)

def to_grayscale(images):
    images = 0.3*images[:,0,...] + 0.58*images[:,1,...] + 0.12*images[:,2,...]
    return torch.unsqueeze(images, 1)

def cpu_squeeze(x):
    if x.device.type == 'cuda':
        x = x.cpu()
    return torch.squeeze(x)

def plot_image_pair(xo, xr, theta=None):
    xo = cpu_squeeze(xo)
    xr = cpu_squeeze(xr)

    fig, ax = plt.subplots(1,2)
    if len(xo.shape) == 3:
        ax[0].imshow(xo.permute(1,2,0))
        ax[1].imshow(xr.permute(1,2,0))
    else:
        ax[0].imshow(xo, cmap='Greys_r')
        ax[1].imshow(xr, cmap='Greys_r')

    if theta:
        ax[0].set_title('orig')
        ax[1].set_title(f'{theta:.3f} radians / {theta*180/math.pi:.1f} degrees')
    plt.show()

def argmax(batch):
    '''
    finds argmax in spatial domain for each image and channel in batch.
    '''

    reshape = batch.reshape((math.prod(batch.shape[:-2]), -1))
    spatial_coords = unravel_indices(torch.argmax(reshape, 1), batch.shape[-2:])
    full_coords = []
    for m, c in enumerate(spatial_coords):
        i = m // batch.shape[1]
        j = m - i*batch.shape[1]
        full_coords.append([i,j,c[0].item(),c[1].item()])
    return full_coords

def plot_bounds_histogram(bounds, deltas=None, bins=None, transform=None,
        params=None, save=False, save_path=None):

    if bins is None:
        nbins = 10
        nsteps = 5
        bins = np.zeros(nbins+1)
        bins[1:nbins//2+1] = np.array([10**(-i//2) for i in range(nbins,
            0,-2)])
        bins[nbins//2+1:] = np.linspace(0.2,1.,nsteps)

    heights, bins = np.histogram(bounds, bins=bins, density=True)
    heights = heights * np.diff(bins)

    plt.title(f'{transform} bounds histogram\nmax value={heights.max():.3f}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('bound')

    plt.grid(alpha=0.5)
    if transform is not None:
        label = "\n".join("{}:{}".format(k, v) for k, v in params.items())
    plt.bar(bins[:-1], heights, align='edge', width=np.diff(bins), label=label)

    plt.legend(loc=(0.2,0.8))
    if save:
        plt.savefig(save_path, dpi=300)
    plt.show()

    if deltas is not None:
        plt.hist(deltas, bins=30, log=True);
        plt.title(f'{transform} top 2 prediction probability difference')
        plt.xlabel('top2 diff')
        plt.show()

def get_filename(dataset, model_type, inf_type, transform, a, z, steps,
        nsamples):
    name_bounds = f'bounds|{dataset}|{model_type}|{inf_type}|{transform}|{nsamples}|{a}-{z}-{steps}.npy'
    name_deltas = f'deltas|{dataset}|{model_type}|{inf_type}|{transform}|{nsamples}|{a}-{z}-{steps}.npy'
    name_hitmask = f'hitmask|{dataset}|{model_type}|{inf_type}|{transform}|{nsamples}|{a}-{z}-{steps}.npy'
    return name_bounds, name_deltas, name_hitmask

def get_log_bins(nbins=10, nsteps=5):
    bins = np.zeros(nbins+1)
    bins[1:nbins//2+1] = np.array([10**(-i//2) for i in range(nbins, 0,-2)])
    bins[nbins//2+1:] = np.linspace(0.2,1.,nsteps)
    return bins
