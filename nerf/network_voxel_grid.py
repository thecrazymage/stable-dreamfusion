import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp, biased_softplus
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize

# taken from https://github.com/sunset1995/DirectVoxGO
class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, volumetric, **kwargs):
        super(DenseGrid, self).__init__()
        self.volumetric = volumetric
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))

    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        if self.volumetric:
            xyz = xyz.reshape(1,1,1,-1,3)
        else:
            xyz = xyz.reshape(1,1,-1,2)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 volume_grid_resolution=[128, 128, 128],
                 background_resolution=[256, 256],
                 xyz_min=[-0.9, -0.9, -0.9],
                 xyz_max=[0.9, 0.9, 0.9],
                 use_finite_difference_normal=False):

        super().__init__(opt)

        self.use_finite_difference_normal = use_finite_difference_normal
        self.voldensity_grid = DenseGrid(1, volume_grid_resolution, xyz_min, xyz_max, True)
        self.density_activation = trunc_exp if self.opt.density_activation == 'exp' else biased_softplus

        self.albedo_grid = DenseGrid(3, volume_grid_resolution, xyz_min, xyz_max, True)
        self.albedo_activation = torch.sigmoid

        if self.opt.bg_radius > 0:
            self.background_grid = DenseGrid(3,
                                             background_resolution,
                                             (0, -torch.pi),
                                             (torch.pi, torch.pi),
                                             False)
            self.background_activation = torch.sigmoid


    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx = epsilon * torch.eye(3, device=x.device)
        dx_pos = self.get_density(x + dx[0]).clamp(-self.bound, self.bound)
        dx_neg = self.get_density(x - dx[0]).clamp(-self.bound, self.bound)
        dy_pos = self.get_density(x + dx[1]).clamp(-self.bound, self.bound)
        dy_neg = self.get_density(x - dx[1]).clamp(-self.bound, self.bound)
        dz_pos = self.get_density(x + dx[2]).clamp(-self.bound, self.bound)
        dz_neg = self.get_density(x - dx[2]).clamp(-self.bound, self.bound)
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal

    def autograd_normal(self, x):
        x.requires_grad_()
        density = self.get_density(x).sum()
        normal = torch.autograd.grad([density],
                                     [x],
                                     retain_graph=True,)[0]
        return -normal

    def normal(self, x):
        if self.use_finite_difference_normal:
            normal = self.finite_difference_normal(x)
        else:
            normal = self.autograd_normal(x)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)
        return normal

    def get_density(self, x):
        h_density = self.voldensity_grid(x)
        density = self.density_activation(h_density + self.density_blob(x))
        return density

    def density(self, x):
        density = self.get_density(x)
        albedo = self.get_albedo(x)
        return {
            'sigma': density,
            'albedo': albedo,
        }

    def get_albedo(self, x):
        h_albedo = self.albedo_grid(x)
        albedo = self.albedo_activation(h_albedo)
        return albedo

    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        sigma = self.get_density(x)
        albedo = self.get_albedo(x)

        if shading == 'albedo':
            normal = None
            color = albedo
        
        else: # lambertian shading

            normal = self.normal(x)

            lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0) # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
            
        return sigma, color, normal

    def background(self, d):
        d_normalized = nn.functional.normalize(d, dim=-1)
        theta = torch.acos(d_normalized[..., 2])
        phi = torch.sign(d_normalized[..., 1]) * torch.acos(d_normalized[..., 0])
        h_background = self.background_grid(torch.stack([theta, phi], dim=-1))
        return self.background_activation(h_background - 5.)

    def get_params(self, lr):
        params = [
                {'params': self.voldensity_grid.parameters(), 'lr': lr},
                {'params': self.albedo_grid.parameters(), 'lr': lr},
        ]

        if self.opt.bg_radius > 0:
            params.append({'params': self.background_grid.parameters(), 'lr': lr})

        return params
