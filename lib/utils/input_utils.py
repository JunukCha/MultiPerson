import torch

import numpy as np
import cv2

from lib.utils.img_utils import convert_cvimg_to_tensor


def get_pose_estimator_input(img_patch, FLAGS):
    img_patch = img_patch.copy()
    img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
    img_patch_resize_256 = cv2.resize(img_patch, (FLAGS.proc_side, FLAGS.proc_side))

    img_pe_input = torch.from_numpy(img_patch_resize_256.copy())
    img_pe_input = img_pe_input.unsqueeze(0)
    img_pe_input = img_pe_input.permute(0, 3, 1, 2).float()
    img_pe_input /= 255
    img_pe_input = img_pe_input.cuda()

    imshape = torch.FloatTensor([FLAGS.proc_side, FLAGS.proc_side])
    fov_degrees = FLAGS.fov_degrees
    fov_radians = fov_degrees * torch.FloatTensor([np.pi / 180])
    larger_size = torch.max(imshape)
    focal_length = larger_size / (torch.tan(fov_radians/2)*2)
    intrinsic = torch.FloatTensor(
        [[
            [focal_length, 0, imshape[1]/2],  
            [0, focal_length, imshape[0]/2],
            [0, 0, 1],
        ]]
    )
    intrinsic = intrinsic.cuda()
    return img_patch_resize_256, img_pe_input, intrinsic

def get_feature_extractor_input(img_patch):
    img_patch = img_patch.copy()
    img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
    img_patch_resize_224 = cv2.resize(img_patch, (224, 224))

    img_fe_input = convert_cvimg_to_tensor(img_patch_resize_224)
    img_fe_input = img_fe_input.unsqueeze(0)
    img_fe_input = img_fe_input.cuda()
    return img_fe_input

def get_ik_input(img_patch, demo_cfg, FLAGS):
    img_patch = img_patch.copy()
    img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
    img_ik_input = cv2.resize(img_patch, (FLAGS.proc_side, FLAGS.proc_side))
    
    img_ik_input = torch.from_numpy(np.transpose(img_ik_input, (2, 0, 1))).float()
    if img_ik_input.max() > 1:
        img_ik_input /= 255
    img_ik_input[0].add_(demo_cfg.InverseKinematic.IMG_MEAN[0])
    img_ik_input[1].add_(demo_cfg.InverseKinematic.IMG_MEAN[1])
    img_ik_input[2].add_(demo_cfg.InverseKinematic.IMG_MEAN[2])

    img_ik_input[0].div_(demo_cfg.InverseKinematic.IMG_STD[0])
    img_ik_input[1].div_(demo_cfg.InverseKinematic.IMG_STD[1])
    img_ik_input[2].div_(demo_cfg.InverseKinematic.IMG_STD[2])
    img_ik_input = img_ik_input.unsqueeze(0)
    img_ik_input = img_ik_input.cuda()
    return img_ik_input


