import os
import os.path as osp

import yaml
from easydict import EasyDict as edict
import pickle

def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config

def read_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data

def make_folder(cfg, imgfile):
    split_images_folder = cfg.SAVE_FOLDER.split_images_folder
    pose_results_folder = cfg.SAVE_FOLDER.pose_results_folder
    mesh_results_folder = cfg.SAVE_FOLDER.mesh_results_folder
    
    base_imgfile = osp.basename(imgfile).split(".")[0]
    split_images_folder = osp.join(split_images_folder, f"{base_imgfile}")
    pose_results_folder = osp.join(pose_results_folder, f"{base_imgfile}")
    mesh_results_folder = osp.join(mesh_results_folder, f"{base_imgfile}")

    os.makedirs(split_images_folder, exist_ok=True)
    os.makedirs(pose_results_folder, exist_ok=True)
    os.makedirs(mesh_results_folder, exist_ok=True)
    return split_images_folder, pose_results_folder, mesh_results_folder