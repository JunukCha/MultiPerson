import torch
import torch.nn as nn

from timm.models.layers import DropPath
import einops
import cv2
import matplotlib.pyplot as plt
import numpy as np
from lib.utils.file_utils import read_pickle

class SiLU(nn.Module):
    @staticmethod 
    def forward(x): 
        return x * torch.sigmoid(x) 

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, g=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, (k - 1) // 2 if p is None else p, 1, g, bias=False)
        self.norm = nn.BatchNorm2d(out_ch, 0.001, 0.01)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    
class SE(torch.nn.Module):
    def __init__(self, ch, r):
        super().__init__()
        self.se = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Conv2d(ch, ch // (4 * r), 1),
                                      SiLU(),
                                      torch.nn.Conv2d(ch // (4 * r), ch, 1),
                                      torch.nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)
    
    
class Residual(nn.Module):
    """
    [https://arxiv.org/pdf/1801.04381.pdf]
    """

    def __init__(self, in_ch, out_ch, s, r, dp_rate=0, fused=True, br=False):
        super().__init__()
        identity = nn.Identity()
        self.add = s == 1 and in_ch == out_ch
        self.fused = fused
        self.br = br

        if fused:
            features = [Conv(in_ch, r * in_ch, activation=SiLU(), k=3, s=s),
                        Conv(r * in_ch, out_ch, identity) if r != 1 else identity,
                        DropPath(dp_rate) if self.add else identity]
        else:
            features = [Conv(in_ch, r * in_ch, SiLU()) if r != 1 else identity,
                        Conv(r * in_ch, r * in_ch, SiLU(), 3, s, r * in_ch),
                        SE(r * in_ch, r), 
                        Conv(r * in_ch, out_ch, identity),
                        DropPath(dp_rate) if self.add else identity]

        self.res = nn.Sequential(*features)

    def forward(self, x):
        x0 = x
        if self.fused:
            x1 = self.res[0](x0)
            x2 = self.res[1](x1)
            xout = self.res[2](x2)
            
        else:
            x1 = self.res[0](x0)
            x2 = self.res[1](x1)
            x3 = self.res[2](x2)
            x4 = self.res[3](x3)
            xout = self.res[4](x4)
        return x + xout if self.add else xout

def init_weight(model):
    import math
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out = fan_out // m.groups
            torch.nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, torch.nn.Linear):
            init_range = 1.0 / math.sqrt(m.weight.size()[0])
            torch.nn.init.uniform_(m.weight, -init_range, init_range)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

class EfficientNet(nn.Module):
    """
     efficientnet-v2-s :
                        num_dep = [2, 4, 4, 6, 9, 15, 0]
                        filters = [24, 48, 64, 128, 160, 256, 256, 1280]
     efficientnet-v2-m :
                        num_dep = [3, 5, 5, 7, 14, 18, 5]
                        filters = [24, 48, 80, 160, 176, 304, 512, 1280]
     efficientnet-v2-l :
                        num_dep = [4, 7, 7, 10, 19, 25, 7]
                        filters = [32, 64, 96, 192, 224, 384, 640, 1280]
    """

    def __init__(self, drop_rate=0, num_class=1000):
        super().__init__()
        num_dep = [4, 7, 7, 10, 19, 25, 7]
        filters = [32, 64, 96, 192, 224, 384, 640, 1280]

        dp_index = 0
        dp_rates = [x.item() for x in torch.linspace(0, 0.2, sum(num_dep))]
        
#         self.stem = Conv(3, filters[0], SiLU(), 3, 2, p=0)
        self.stem = Conv(3, filters[0], SiLU(), 3, 2)
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        for i in range(num_dep[0]):
            if i == 0:
#                 self.p1.append(Conv(3, filters[0], SiLU(), 3, 2))
                self.p1.append(Residual(filters[0], filters[0], 1, 1, dp_rates[dp_index]))
            else:
                self.p1.append(Residual(filters[0], filters[0], 1, 1, dp_rates[dp_index]))
            dp_index += 1
        # p2/4
        for i in range(num_dep[1]):
            if i == 0:
                self.p2.append(Residual(filters[0], filters[1], 2, 4, dp_rates[dp_index]))
            else:
                self.p2.append(Residual(filters[1], filters[1], 1, 4, dp_rates[dp_index]))
            dp_index += 1
        # p3/8
        for i in range(num_dep[2]):
            if i == 0:
                self.p3.append(Residual(filters[1], filters[2], 2, 4, dp_rates[dp_index]))
            else:
                self.p3.append(Residual(filters[2], filters[2], 1, 4, dp_rates[dp_index]))
            dp_index += 1
        # p4/16
        for i in range(num_dep[3]):
            if i == 0:
                self.p4.append(Residual(filters[2], filters[3], 2, 4, dp_rates[dp_index], False))
            else:
                self.p4.append(Residual(filters[3], filters[3], 1, 4, dp_rates[dp_index], False))
            dp_index += 1
        for i in range(num_dep[4]):
            if i == 0:
                self.p4.append(Residual(filters[3], filters[4], 1, 6, dp_rates[dp_index], False))
            else:
                self.p4.append(Residual(filters[4], filters[4], 1, 6, dp_rates[dp_index], False))
            dp_index += 1
        # p5/32
        for i in range(num_dep[5]):
            if i == 0:
                self.p5.append(Residual(filters[4], filters[5], 2, 6, dp_rates[dp_index], False, True))
            else:
                self.p5.append(Residual(filters[5], filters[5], 1, 6, dp_rates[dp_index], False))
            dp_index += 1
        for i in range(num_dep[6]):
            if i == 0:
                self.p5.append(Residual(filters[5], filters[6], 1, 6, dp_rates[dp_index], False))
            else:
                self.p5.append(Residual(filters[6], filters[6], 1, 6, dp_rates[dp_index], False))
            dp_index += 1

        self.p1 = nn.Sequential(*self.p1)
        self.p2 = nn.Sequential(*self.p2)
        self.p3 = nn.Sequential(*self.p3)
        self.p4 = nn.Sequential(*self.p4)
        self.p5 = nn.Sequential(*self.p5)

        self.head = Conv(filters[6], filters[7], SiLU())
#         self.fc2 = nn.Linear(filters[7], num_class)

        self.drop_rate = drop_rate

        init_weight(self)

    def forward(self, x):
#         x = self.stem(fixed_padding(x, 3))
        x = self.stem(x)
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)

        return self.head(x)

class MetrabsHeads(nn.Module):
    def __init__(self, cfg, n_points):
        super().__init__()
        self.n_points = n_points
        self.n_outs = [self.n_points, 8*self.n_points]
        self.conv_final = nn.Conv2d(1280, sum(self.n_outs), 1)
        self.FLAGS = cfg.PoseEstimator.FLAGS
        
    def forward(self, x):
        x = self.conv_final(x)
        logits2d, logits3d = torch.split(x, self.n_outs, dim=1)
        logits2d = logits2d.permute(0, 2, 3, 1) # B H W C
        logits3d = logits3d.permute(0, 2, 3, 1) # B H W C
        b, h, w, c = logits3d.shape
        logits3d = logits3d.reshape(b, h, w, 8, self.n_points)
        coords3d = self.soft_argmax(logits3d.to(torch.float32), dim=[2, 1, 3]) # 2: width -> 3, 1: height -> 2, 3: Channel -> 1
        coords3d_rel_pred = self.heatmap_to_metric(coords3d)
        coords2d = self.soft_argmax(logits2d.to(torch.float32), dim=[2, 1])
        coords2d_pred = self.heatmap_to_image(coords2d)
        return coords2d_pred, coords3d_rel_pred

    def soft_argmax(self, inp, dim):
        return self.decode_heatmap(self.softmax(inp, dim=dim), dim=dim)

    def softmax(self, target, dim=-1):
        max_along_axis = torch.amax(target, dim=dim, keepdim=True)
        exponentiated = torch.exp(target - max_along_axis)
        denominator = torch.sum(exponentiated, dim=dim, keepdim=True)
        return exponentiated / denominator

    def decode_heatmap(self, inp, dim, output_coord_axis=-1):
        if not isinstance(dim, (tuple, list)):
            dim = [dim]
        
        heatmap_axes = dim
        
        result = []
        for ax in heatmap_axes:
            other_heatmap_axes = [other_ax for other_ax in heatmap_axes if other_ax != ax]
            summed_over_other_heatmap_axes = torch.sum(inp, axis=other_heatmap_axes, keepdims=True)
            coords = (torch.linspace(0.0, 1.0, inp.shape[ax])).to(inp.dtype).to(summed_over_other_heatmap_axes.device)
            decoded = torch.tensordot(summed_over_other_heatmap_axes, coords, dims=[[ax], [0]])
            decoded = torch.unsqueeze(decoded, ax)
            for _ in range(len(dim)):
                decoded = torch.squeeze(decoded, dim=1)
            result.append(decoded)
        return torch.stack(result, dim=output_coord_axis)

    def heatmap_to_image(self, coords):
        stride = self.FLAGS["stride"]
        stride //= self.FLAGS["final_transposed_conv"]
        last_image_pixel = self.FLAGS["proc_side"] - 1
        last_receptive_center = last_image_pixel - (last_image_pixel % stride)
        coords_out = coords * last_receptive_center
        if self.FLAGS["centered_stride"]:
            coords_out = coords_out + stride // 2

        return coords_out

    def heatmap_to_metric(self, coords):
        coords2d = self.heatmap_to_image(
            coords[..., :2]) * self.FLAGS["box_size"] / self.FLAGS["proc_side"]
        return torch.cat([coords2d, coords[..., 2:] * self.FLAGS["box_size"]], dim=-1)


class PoseEstimator(nn.Module):
    def __init__(self, cfg, backbone):
        super().__init__()
        self.backbone = backbone
        n_raw_points = 32
        self.heatmap_heads = MetrabsHeads(cfg, n_points=n_raw_points)
        
        self.FLAGS = cfg.PoseEstimator.FLAGS
        to_122_file = cfg.PoseEstimator.to_122_file
        self.to_122 = torch.FloatTensor(np.load(to_122_file)).cuda()

        self.joint_info = read_pickle(cfg.PoseEstimator.joint_info_file)
        
        
    def forward(self, x, new_intrinsics, default_intrinsics, skeleton="smpl_24"):
        x = self.backbone(x)
        coords2d, coords3d = self.heatmap_heads(x)
        coords2d = torch.einsum('bjc,jJ->bJc', coords2d, self.to_122)
        coords3d = torch.einsum('bjc,jJ->bJc', coords3d, self.to_122)
        coords3d_abs = self.reconstruct_absolute(coords2d, coords3d, new_intrinsics)
        
        poses2d_flat_normalized = to_homogeneous(project(coords3d_abs))
        poses2d_flat = torch.einsum('bnk,bjk->bnj', poses2d_flat_normalized,
                                     default_intrinsics[..., :2, :])
        
        skeleton_indices = self.joint_info["per_skeleton_indices"][skeleton]
        edges = self.joint_info["per_skeleton_joint_edges"][skeleton]

        poses2d_flat_sampled = poses2d_flat[:, skeleton_indices]
        coords3d_sampled = coords3d[:, skeleton_indices]
        coords3d_abs_sampled = coords3d_abs[:, skeleton_indices]
        return poses2d_flat_sampled, coords3d_sampled, coords3d_abs_sampled, skeleton_indices, edges
    
    def reconstruct_absolute(self, coords2d, coords3d_rel, intrinsics):
        inv_intrinsics = torch.linalg.inv(intrinsics.to(coords2d.dtype))
        coords2d_normalized = torch.matmul(
            to_homogeneous(coords2d), inv_intrinsics.transpose(1, 2))[..., :2]
        
        is_predicted_to_be_in_fov = self.is_within_fov(coords2d)
        
        ref = self.reconstruct_ref_fullpersp(coords2d_normalized, coords3d_rel, is_predicted_to_be_in_fov)
        
        coords_abs_3d_based = coords3d_rel + torch.unsqueeze(ref, 1)
        reference_depth = ref[:, 2]
        relative_depths = coords3d_rel[..., 2]
        
        coords_abs_2d_based = back_project(coords2d_normalized, relative_depths, reference_depth)
        return torch.where(
            is_predicted_to_be_in_fov[..., np.newaxis], coords_abs_2d_based, coords_abs_3d_based)
        
    def is_within_fov(self, imcoords):
        stride_train = self.FLAGS["stride"] / self.FLAGS["final_transposed_conv"]
        offset = -stride_train / 2 if not self.FLAGS["centered_stride"] else 0
        lower = torch.FloatTensor([stride_train * 0.75 + offset]).to(imcoords.device)
        upper = torch.FloatTensor([self.FLAGS["proc_side"] - stride_train * 0.75 + offset]).to(imcoords.device)
        return torch.all(torch.logical_and(imcoords >= lower, imcoords <= upper), dim=-1)

    def reconstruct_ref_fullpersp(self, normalized_2d, coords3d_rel, validity_mask):
        def rms_normalize(x):
            scale = torch.sqrt(torch.mean(torch.square(x)))
            normalized = x / scale
            return scale, normalized

        n_batch = normalized_2d.shape[0]
        n_points = normalized_2d.shape[1]
        eyes = torch.tile(torch.unsqueeze(torch.eye(2, 2), 0), [n_batch, n_points, 1]).to(normalized_2d.device)
        scale2d, reshaped2d = rms_normalize(normalized_2d.reshape(-1, n_points * 2, 1))
        A = torch.cat((eyes, -reshaped2d), dim=2)
        rel_backproj = normalized_2d * coords3d_rel[:, :, 2:] - coords3d_rel[:, :, :2]
        scale_rel_backproj, b = rms_normalize(rel_backproj.reshape(-1, n_points * 2, 1))

        weights = validity_mask.to(torch.float32) + np.float32(1e-4)
        weights = einops.repeat(weights, 'b j -> b (j c) 1', c=2)

        ref = torch.linalg.lstsq(A * weights, b * weights).solution
        ref = torch.cat((ref[:, :2], ref[:, 2:] / scale2d), dim=1) * scale_rel_backproj
        return torch.squeeze(ref, dim=-1)

def project(points):
    return points[..., :2] / points[..., 2:3]

def back_project(camcoords2d, delta_z, z_offset):
    return to_homogeneous(camcoords2d) * torch.unsqueeze(delta_z + torch.unsqueeze(z_offset, -1), -1)

def to_homogeneous(x):
    return torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)

