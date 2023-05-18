import sys
sys.path.append("YOLOv4")
import os

import torch

import cv2
import numpy as np
import argparse

from YOLOv4.tool.utils import load_class_names, split_boxes_cv2
from YOLOv4.tool.torch_utils import do_detect
from lib.utils.model_utils import create_all_network
from lib.utils.input_utils import (
    get_pose_estimator_input,
    get_feature_extractor_input,
    get_ik_input
)
from lib.utils.renderer import Renderer
from lib.utils.file_utils import (
    update_config, 
    make_folder
)
from lib.utils.output_utils import (
    save_3d_joints,
    save_2d_joints,
    process_output, 
    save_mesh_obj,
    save_mesh_rendering, 
    save_mesh_pkl,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default="demo_image/demo1.jpg")
    parser.add_argument("--demo_cfg", default="configs/demo.yaml")
    args = parser.parse_args()
    assert os.path.exists(args.img), "You should use the image that exists."
    demo_cfg = update_config(args.demo_cfg)
    
    namesfile = demo_cfg.YOLO.namesfile
    height = demo_cfg.YOLO.target_height
    width = demo_cfg.YOLO.target_width
    n_classes = demo_cfg.YOLO.n_classes
    imgfile = args.img

    FLAGS = demo_cfg.PoseEstimator.FLAGS
    max_num_person = demo_cfg.SmplTR.max_num_person
    
    yolo, pose_estimator, ik_net, \
    feature_extractor, smpl_layer, smplTR \
        = create_all_network(demo_cfg)
    
    split_images_folder, pose_results_folder, \
    mesh_results_folder = make_folder(demo_cfg, imgfile)

    orig_img = cv2.imread(imgfile)
    orig_height, orig_width = orig_img.shape[:2]
    renderer = Renderer(smpl=smpl_layer, resolution=(orig_width, orig_height), orig_img=True)

    # Inference input size is 416*416 does not mean training size is the same
    # Training size could be 608*608 or even other sizes
    # Optional inference sizes:
    #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
    #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
    yolo_input_img = cv2.resize(orig_img, (width, height))
    yolo_input_img = cv2.cvtColor(yolo_input_img, cv2.COLOR_BGR2RGB)

    for i in range(2):  # This 'for' loop is for speed check
                        # Because the first iteration is usually longer
        boxes = do_detect(yolo, yolo_input_img, 0.4, 0.6, use_cuda=True)

    class_names = load_class_names(namesfile)
    img_patch_list, refined_boxes, trans_invs = split_boxes_cv2(orig_img, boxes[0], split_images_folder, class_names)
    refined_boxes = np.array(refined_boxes)

    num_person = len(img_patch_list)
    num_person = min(num_person, max_num_person)
    
    feature_dump = torch.zeros(1, max_num_person, 2048).float().cuda()
    rot6d_dump = torch.zeros(1, max_num_person, 24, 6).float().cuda()
    betas_dump = torch.zeros(1, max_num_person, 10).float().cuda()
    
    j2ds_pixel = []
    for person_id, img_patch in enumerate(img_patch_list[:num_person]):
        img_plot, img_pe_input, intrinsic = get_pose_estimator_input(img_patch, FLAGS)
        
        with torch.no_grad():
            j2d, j3d, j3d_abs, skeleton_indices, edges \
                = pose_estimator(img_pe_input, intrinsic, intrinsic)
        save_3d_joints(j3d_abs, edges, pose_results_folder, person_id)
        save_2d_joints(img_plot, j2d, edges, pose_results_folder, person_id)
        
        img_ik_input = get_ik_input(img_patch, demo_cfg, FLAGS)
        j3ds_abs_meter = j3d_abs / 1000
        ik_net_output = ik_net(img_ik_input, j3ds_abs_meter)
        rot6d_ik_net = ik_net_output.pred_rot6d
        betas_ik_net = ik_net_output.pred_shape

        img_fe_input = get_feature_extractor_input(img_patch)
        
        img_feature = feature_extractor.extract(img_fe_input)
        feature_dump[0][person_id] = img_feature[0]
        rot6d_dump[0][person_id] = rot6d_ik_net[0]
        betas_dump[0][person_id] = betas_ik_net[0]
    
    with torch.no_grad():
        refined_rot6d, refined_betas, refined_cam = smplTR(feature_dump, rot6d_dump, betas_dump)
        axis_angle, rot6d, betas, cam, verts, faces \
            = process_output(smpl_layer, refined_rot6d, refined_betas, refined_cam)
    
    save_mesh_obj(verts, faces, mesh_results_folder)
    save_mesh_rendering(renderer, verts, refined_boxes, cam, orig_height, orig_width, mesh_results_folder)
    save_mesh_pkl(axis_angle, betas, cam, mesh_results_folder)