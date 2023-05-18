import torch
from torchvision import models

from YOLOv4.models import Yolov4
from lib.models.pose_estimator import EfficientNet, PoseEstimator
from lib.models.feature_extractor import Bottleneck, FeatureExtractor
from lib.models.transformer import SmplTR
from lib.models import builder
from lib.models.smpl import create_smpl
from lib.utils.file_utils import update_config

def create_YOLO(cfg):
    n_classes = cfg.YOLO.n_classes
    weight_file = cfg.YOLO.weight_file

    yolo = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)
    pretrained_dict = torch.load(weight_file)
    yolo = yolo.cuda()
    yolo.load_state_dict(pretrained_dict)
    yolo.eval()
    return yolo

def create_pose_estimator(cfg):
    weight_file = cfg.PoseEstimator.weight_file

    backbone = EfficientNet()
    backbone = backbone.cuda()
    pose_estimator = PoseEstimator(cfg, backbone)
    pretrained_dict = torch.load(weight_file)
    pose_estimator = pose_estimator.cuda()
    pose_estimator.load_state_dict(pretrained_dict)
    pose_estimator.eval()
    return pose_estimator

def create_inverse_kinematics(cfg):
    ik_cfg = cfg.InverseKinematic.cfg
    weight_file = cfg.InverseKinematic.weight_file

    ik_cfg = update_config(ik_cfg)
    ik_net = builder.build_sppe(ik_cfg.MODEL)
    pretrained_dict = torch.load(weight_file)
    ik_net = ik_net.cuda()
    ik_net.load_state_dict(pretrained_dict, strict=False)
    ik_net.eval()
    return ik_net

def create_feature_extractor(cfg):
    weight_file = cfg.FeatureExtractor.weight_file

    feature_extractor = FeatureExtractor(Bottleneck, [3, 4, 6, 3])
    pretrained_dict = torch.load(weight_file)
    feature_extractor = feature_extractor.cuda()
    feature_extractor.load_state_dict(pretrained_dict['model'], strict=False)
    feature_extractor.eval()
    return feature_extractor

def create_smpl_transformer(cfg):
    weight_file = cfg.SmplTR.weight_file
    smplTR = SmplTR(cfg).cuda()
    pretrained_dict = torch.load(weight_file)
    smplTR = smplTR.cuda()
    smplTR.load_state_dict(pretrained_dict["model"], strict=False)
    smplTR.eval()
    return smplTR

def create_all_network(cfg):
    yolo = create_YOLO(cfg)
    pose_estimator = create_pose_estimator(cfg)
    ik_net = create_inverse_kinematics(cfg)
    feature_extractor = create_feature_extractor(cfg)
    smpl_layer = create_smpl(model_path=cfg.SMPL.model_path)
    smplTR = create_smpl_transformer(cfg)
    return (
        yolo,
        pose_estimator,
        ik_net,
        feature_extractor,
        smpl_layer,
        smplTR,
    )