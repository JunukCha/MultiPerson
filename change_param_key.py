import torch

import argparse

from lib.utils.file_utils import update_config

def change_key_name_neek_to_neck(cfg):
    weight_file = cfg.YOLO.weight_file
    pretrained_dict = torch.load(weight_file)
    new_state_dict = {}
    for key, value in pretrained_dict.items():
        if "neek" in key:
            new_key = key.replace("neek", "neck")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    torch.save(
        new_state_dict,
        "YOLOv4/weight/new_yolov4.pth"
    )
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_cfg", default="configs/demo.yaml")
    args = parser.parse_args()
    demo_cfg = update_config(args.demo_cfg)
    change_key_name_neek_to_neck(demo_cfg)

