import os.path as osp
import trimesh
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from lib.utils.img_utils import convert_crop_cam_to_orig_img
from lib.utils.pose_utils import (
    matrix_to_axis_angle,
    rot6d_to_rotmat,
)

def process_output(smpl_layer, rot6d, betas, cam):
    rot6d = rot6d.reshape(-1, 144)
    betas = betas.reshape(-1, 10)
    cam = cam.reshape(-1, 3)
    rotmat = rot6d_to_rotmat(rot6d)
    rotmat = rotmat.reshape(-1, 24, 3, 3)
    axis_angle = matrix_to_axis_angle(rotmat)
    axis_angle = axis_angle.reshape(-1, 24*3)
    smpl_output_est = smpl_layer(poses=axis_angle, betas=betas)

    verts = smpl_output_est["verts"].cpu().numpy()
    faces = smpl_layer.faces_tensor.cpu().numpy()

    return axis_angle, rot6d, betas, cam, verts, faces

def save_mesh_obj(verts, faces, num_person, mesh_results_folder):
    for person_id, vert in enumerate(verts[:num_person]):
        mesh = trimesh.Trimesh(vert, faces)
        mesh.export(osp.join(mesh_results_folder, f"mesh{person_id}.obj"))

def save_mesh_rendering(renderer, verts, boxes, cam, orig_height, orig_width, num_person, mesh_results_folder):
    orig_img = np.ones((orig_height, orig_width, 3))*255
    render_img = None
    
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_person + 2)]
    colors = [(c[2], c[1], c[0]) for c in colors]
    for person_id in range(num_person):
        orig_cam = convert_crop_cam_to_orig_img(
            cam=cam[person_id:person_id+1].detach().cpu().numpy(),
            bbox=boxes[person_id:person_id+1],
            img_width=orig_width,
            img_height=orig_height
        )
        if render_img is None:
            render_img = renderer.render(
                orig_img,
                verts[person_id],
                cam=orig_cam[0],
                color=colors[person_id],
            )
        else:
            render_img = renderer.render(
                render_img,
                verts[person_id],
                cam=orig_cam[0],
                color=colors[person_id],
            )
    cv2.imwrite(osp.join(mesh_results_folder, f"mesh.jpg"), render_img)

def save_mesh_pkl(axis_angle, betas, cam, num_person, mesh_results_folder):
    for person_id in range(num_person):
        data = {
            "thetas": axis_angle[person_id].detach().cpu().numpy(),
            "betas": betas[person_id].detach().cpu().numpy(),
            "cam": cam[person_id].detach().cpu().numpy()
        }
        with open(osp.join(mesh_results_folder, f"smpl_{person_id}.pkl"), "wb") as f:
            pickle.dump(data, f)

def save_3d_joints(j3d, edges, pose_results_folder, person_id):
    j3d = j3d[0].cpu().numpy()
    j3d[:, 1], j3d[:, 2] = j3d[:, 2], -j3d[:, 1]

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(edges) + 2)]
    colors = [(c[2], c[1], c[0]) for c in colors]
    fig = plt.figure()
    pose_ax = fig.add_subplot(1, 1, 1, projection='3d')
    pose_ax.view_init(5, -85)
    pose_ax.set_xlim3d(-1500, 1500)
    pose_ax.set_zlim3d(-1500, 1500)
    pose_ax.set_ylim3d(0, 3000)
    
    for j in j3d:
        pose_ax.scatter(j[0], j[1], j[2], c='r', s=2)

    for l, edge in enumerate(edges):
        pose_ax.plot(
            [j3d[edge[0]][0], j3d[edge[1]][0]], 
            [j3d[edge[0]][1], j3d[edge[1]][1]], 
            [j3d[edge[0]][2], j3d[edge[1]][2]], 
            c=colors[l]
        )

    plt.savefig(osp.join(pose_results_folder, f"image{person_id}_3d.jpg"))

def save_2d_joints(img, j2d, edges, pose_results_folder, person_id):
    j2d = j2d[0].cpu().numpy()
    
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(edges) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    for j in j2d:
        cv2.circle(img, (int(j[0]), int(j[1])), 2, (0, 0, 255), -1)
    
    for l, edge in enumerate(edges):
        cv2.line(img,
            [int(j2d[edge[0]][0]), int(j2d[edge[0]][1])], 
            [int(j2d[edge[1]][0]), int(j2d[edge[1]][1])], 
            colors[l], 
            2
        )
    cv2.imwrite(osp.join(pose_results_folder, f"image{person_id}_2d.jpg"), img[..., ::-1])