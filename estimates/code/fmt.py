import math
import numpy as np

import torch
import torch.fft as tfft

from .utils import argmax, map_coordinates


def fft_shift(x, axes=None):
    '''
    Shift the zero-frequency component to the center of the spectrum.
    '''
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, (np.integer, int)):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[ax] // 2 for ax in axes]
    return torch.roll(x, shift, axes)

def fft_shift_mag(image):
    '''
    Calculate shifted fft of `image` and its magnitude.
    '''
    fft = tfft.fft2(image)
    fft_shifted = fft_shift(fft, (-2,-1))
    mag = torch.abs(fft_shifted)
    return fft_shifted, mag

def hpf(image):
    '''
    Construct high pass filter.
    '''
    pi2 = math.pi / 2.0
    rows = torch.cos(torch.linspace(-pi2, pi2, image.shape[-2], device=image.device))
    cols = torch.cos(torch.linspace(-pi2, pi2, image.shape[-1], device=image.device))
    x = torch.outer(rows, cols)
    return (1.0 - x) * (2.0 - x)

def log_polar_params(image):
    '''
    Compute parameters for log polar transformation.
    '''
    # TODO: generic image size - can do faster for square ones
    image_shape = torch.tensor(image.shape[-2:])
    center_trans = [torch.floor((image_shape[0] + 1) / 2),
                    torch.floor((image_shape[1] + 1 ) / 2)]
    maxdiff = torch.maximum(torch.tensor(center_trans),
                            image_shape - torch.tensor(center_trans))
    maxdist = torch.norm(maxdiff)
    dims_logpolar = [image_shape[0], image_shape[1]]
    log_base = torch.exp(torch.log(maxdist) / dims_logpolar[1])
    angle_step = ( 1.0 * math.pi ) / dims_logpolar[0]
    return center_trans, angle_step, log_base

def convert_log_polar(image, center_trans, angle_step, log_base, mode="nearest"):
    '''
    Transform `image` to log polar representation.
    '''
#     if mode == "nearest":
    logpolar_image = torch.zeros(image.shape, dtype=image.dtype, device=image.device)
    for radius in range(image.shape[-1]):
        act_radius = log_base ** radius
        for angle in range(image.shape[-2]):
            angle_pi = angle * angle_step
            row = int(center_trans[0] + act_radius * torch.sin(angle_pi))
            col = int(center_trans[1] + act_radius * torch.cos(angle_pi))
            if 0 <= row < image.shape[-2] and 0 <= col < image.shape[-1]:
                logpolar_image[..., angle, radius] = image[..., row, col]
    return logpolar_image
# other mode is not yet working, need to debug map_coordinates first
#     else:
#         angles_map = torch.zeros(image.shape[-2:], dtype=torch.float64)
#         angles_vector = -torch.linspace(0, np.pi, image.shape[-2])
#         angles_map.T[:] = angles_vector
#         radius_map = torch.zeros(image.shape[-2:], dtype=torch.float64)
#         radius_vector = torch.pow(log_base,
#                                   torch.arange(image.shape[-1], dtype=torch.float64)) - 1.
#         radius_map[:] = radius_vector
#         x = radius_map * torch.sin(angles_map) + center_trans[1]
#         y = radius_map * torch.cos(angles_map) + center_trans[0]
#         logpolar_image = map_coordinates(image, torch.stack([x, y]))
#         return logpolar_image

def phase_corr(image_orig, image_tran):
    '''
    Compute phase correlation.
    '''
    orig_conj = torch.conj(image_orig)
    result = orig_conj * image_tran
    result /= torch.abs(result)
    rifft = tfft.ifft2(result)
    rmag = torch.abs(rifft)
    coordinates = argmax(rmag)

    return coordinates, rifft

def get_angles(xo, xr):
    '''
    Compute angles and peak image.
    '''
    hpfilter = hpf(xo)
    ohpf = xo * hpfilter[None, :, :]
    rhpf = xr * hpfilter[None, :, :]

    # 3 log polar transform
    center_trans, theta_step, log_base = log_polar_params(ohpf)
    ologpolar = convert_log_polar(ohpf, center_trans, theta_step, log_base)
    rlogpolar = convert_log_polar(rhpf, center_trans, theta_step, log_base)

    # 4 fft(logpolar)
    ologpolar_fft, _ = fft_shift_mag(ologpolar)
    rlogpolar_fft, _ = fft_shift_mag(rlogpolar)

    # 5 phase correlation for rotation angle and scale factor
    coords, rifft = phase_corr(ologpolar_fft, rlogpolar_fft)
    return coords, rifft, theta_step, ologpolar_fft, rlogpolar_fft

def get_mag(x):
    '''
    Compute magnitude invariant.
    '''
    hpfilter = hpf(x)
    xhpf = x * hpfilter[None, :, :]
    center_trans, theta_step, log_base = log_polar_params(xhpf)
    xlogpolar = convert_log_polar(xhpf, center_trans, theta_step, log_base)
    xlogpolar_fft, _ = fft_shift_mag(xlogpolar)

    xmag = torch.abs(xlogpolar_fft)
    return xmag

def rescale(mag):
    shape = mag.shape
    reshape = mag.reshape((math.prod(shape[:-2]), -1))
    min_values = torch.min(reshape,1)[0]
    min_values = min_values.reshape(shape[:-2])
    mag = mag - min_values[:,:,None,None]
    reshape = mag.reshape((math.prod(shape[:-2]), -1))

    max_values = torch.max(reshape,1)[0]
    max_values = max_values.reshape(shape[:-2])

    scaled = mag / max_values[:,:,None,None] * 255
    return scaled

def get_angles_xy(xo, xt):
    '''
    Compute angles, scale factors and translation coordinates.
    TODO not finished yet.
    '''
    offt_shifted, omag = fft_shift_mag(xo)
    rfft_shifted, rmag = fft_shift_mag(xr)

    # 2 high pass filter
    hpfilter = hpf(omag)
    omag_hpf = omag * hpfilter[None, :, :]
    rmag_hpf = rmag * hpfilter[None, :, :]

    # 3 log polar transform
    center_trans, theta_step, log_base = log_polar_params(omag_hpf)

    ologpolar = convert_log_polar(omag_hpf, center_trans, theta_step, log_base)
    rlogpolar = convert_log_polar(rmag_hpf, center_trans, theta_step, log_base)

    # 4 fft(logpolar)
    ologpolar_fft = tfft.fft2(ologpolar)
    rlogpolar_fft = tfft.fft2(rlogpolar)

    # 5 phase correlation for rotation angle and scale factor
    coords, rifft = phase_corr(ologpolar_fft, rlogpolar_fft)

    # 6 rotate and scale back
    # 7 recalc fft on transformed
    # 8 phase correlate for translation coords
    return coords, rifft, theta_step, ologpolar_fft, rlogpolar_fft
