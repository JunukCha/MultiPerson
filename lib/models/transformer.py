import numpy as np

import torch
import torch.nn as nn

from einops import rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim,)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table    


class SmplTR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.SmplTR.dim
        depth = cfg.SmplTR.depth
        heads = cfg.SmplTR.heads
        mlp_dim = cfg.SmplTR.mlp_dim
        dim_head = cfg.SmplTR.dim_head
        dropout = cfg.SmplTR.dropout
        max_num_person = cfg.SmplTR.max_num_person

        smpl_mean_params = cfg.SMPL.smpl_mean_params
        mean_params = np.load(smpl_mean_params)
        self.register_buffer(
            "init_cam",
            torch.from_numpy(mean_params['cam']).unsqueeze(0)
        )

        num_pose = max_num_person*24
        self.max_num_person = max_num_person

        self.pos_embedding = nn.Parameter(torch.randn(1, num_pose, dim))

        pos_encoding = get_sinusoid_encoding_table(num_pose, dim)
        pos_encoding = torch.FloatTensor(pos_encoding)
        self.nn_pos = nn.Embedding.from_pretrained(pos_encoding, freeze=True)

        self.feature_tokens = nn.Linear(1, 24)
        self.betas_tokens = nn.Linear(1, 24)
        self.cam_tokens = nn.Linear(1, 24)
        
        self.feature_embbeding = nn.Linear(2048+10+3+6, dim)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head_pose = nn.Sequential(
            nn.Linear(dim, 6)
        )
        
        self.mlp_head_betas1 = nn.Sequential(
            nn.Linear(dim, dim//2)
        )
        self.mlp_head_betas2 = nn.Sequential(
            nn.Linear(dim//2, 10)
        )

        self.mlp_head_cam1 = nn.Sequential(
            nn.Linear(dim, dim//2)
        )
        self.mlp_head_cam2 = nn.Sequential(
            nn.Linear(dim//2, 3)
        )

        self.betas_detokens = nn.Linear(24, 1)
        self.cam_detokens = nn.Linear(24, 1)

        self.avg_pool = nn.AvgPool1d(24)

    def forward(self, feature, rot6d_ik_net, betas_ik_net):
        pred_cam = self.init_cam.expand(1, self.max_num_person, -1)
        batch_size = feature.shape[0]

        if self.max_num_person == 1:
            num_person = 1
        elif self.max_num_person > 1:
            num_person = feature.shape[1]

        # B: batch, N: num_person
        # feature B N 2048 -> BN 2048
        # betas_ik_net B N 10 -> BN 10
        # pred_cam B N 3 -> BN 3
        feature = feature.reshape(batch_size*num_person, -1)
        betas_ik_net = betas_ik_net.reshape(batch_size*num_person, -1)
        pred_cam = pred_cam.reshape(batch_size*num_person, -1)
        rot6d_ik_net = rot6d_ik_net.reshape(batch_size*num_person, 24, 6)

        # 24: num_joints
        # feature BN 2048 1 -> BN 24 2048
        # betas_ik_net BN 10 1 -> BN 24 10
        # pred_cam BN 3 1 -> BN 24 3
        feature_tokens = self.feature_tokens(feature.unsqueeze(2)).permute(0, 2, 1)
        betas_tokens = self.betas_tokens(betas_ik_net.unsqueeze(2)).permute(0, 2, 1)
        cam_tokens = self.cam_tokens(pred_cam.unsqueeze(2)).permute(0, 2, 1)

        # feature BN 24 2048 -> B N*24 2048
        # rot6d_ik_net B N 24 6 -> B N*24 6
        # betas BN 24 10 -> B N*24 10
        # cam BN 24 3 -> B N*24 3
        feature_tokens = feature_tokens.reshape(batch_size, -1, 2048)
        rot6d_ik_net = rot6d_ik_net.reshape(batch_size, -1, 6)
        betas_tokens = betas_tokens.reshape(batch_size, -1, 10)
        cam_tokens = cam_tokens.reshape(batch_size, -1, 3)

        # transformer_inp B N*24 2048+6+10+3
        transformer_inp = torch.cat([feature_tokens, rot6d_ik_net, betas_tokens, cam_tokens], dim=2)
        transformer_inp = self.feature_embbeding(transformer_inp)
        transformer_opt = self.transformer(transformer_inp)
        # transformer_opt B N*24 2048+6+10+3
        
        # refined_rot6d B N*24 6
        # refined_betas B N*24 10
        # refined_cam B N*24 3
        refined_rot6d = self.mlp_head_pose(transformer_opt)
        refined_betas = self.mlp_head_betas1(transformer_opt)
        refined_betas = self.mlp_head_betas2(refined_betas)
        refined_cam = self.mlp_head_cam1(transformer_opt)
        refined_cam = self.mlp_head_cam2(refined_cam)

        # refined_rot6d B N*24 6 -> B N 24*6
        # refined_betas B N*24 10 -> BN 24 10
        # refined_cam B N*24 3 -> BN 24 3
        refined_rot6d = refined_rot6d.reshape(batch_size, num_person, 24*6)
        refined_betas = refined_betas.reshape(batch_size*num_person, 24, 10)
        refined_cam = refined_cam.reshape(batch_size*num_person, 24, 3)
        
        # refined_betas BN 24 10 -> BN 10 1
        # refined_cam BN 24 3 -> BN 3 1
        refined_betas = self.betas_detokens(refined_betas.permute(0, 2, 1))
        refined_cam = self.cam_detokens(refined_cam.permute(0, 2, 1))

        # refined_betas BN 1 10 -> B N 10
        # refined_cam BN 1 3 -> B N 3
        
        refined_betas = refined_betas.reshape(batch_size, num_person, 10)
        refined_cam = refined_cam.reshape(batch_size, num_person, 3)
        
        # rot6d_ik_net B N*24 6 -> B N 24*6
        # refined_rot6d B N 24*6
        rot6d_ik_net = rot6d_ik_net.reshape(batch_size, num_person, 24*6)
        refined_rot6d = rot6d_ik_net+refined_rot6d

        # betas_ik_net BN 10 -> B N 10
        # refined_betas B N 10
        betas_ik_net = betas_ik_net.reshape(batch_size, num_person, 10)
        refined_betas = betas_ik_net+refined_betas

        # pred_cam BN 1 3 -> B N 3
        # refined_cam B N 3
        pred_cam = pred_cam.reshape(batch_size, num_person, 3)
        refined_cam = pred_cam+refined_cam
        return refined_rot6d, refined_betas, refined_cam