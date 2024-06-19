import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST, DEFAULT_IMAGE_TOKEN, TEXT_SHORT_QUESTION_LIST
import sys, os
from ..model.smpl.rotation_conversions import matrix_to_axis_angle, matrix_to_euler_angles, euler_angles_to_matrix
import roma

def rotvec_to_eulerangles(x):
    x_rotmat = roma.rotvec_to_rotmat(x)
    thetax = torch.atan2(x_rotmat[:,2,1], x_rotmat[:,2,2])
    thetay = torch.atan2(-x_rotmat[:,2,0], torch.sqrt(x_rotmat[:,2,1]**2+x_rotmat[:,2,2]**2))
    thetaz = torch.atan2(x_rotmat[:,1,0], x_rotmat[:,0,0])
    return thetax, thetay, thetaz

def eulerangles_to_rotmat(thetax, thetay, thetaz):
    N = thetax.numel()
    # rotx
    rotx = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
    roty = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
    rotz = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
    rotx[:,1,1] = torch.cos(thetax)
    rotx[:,2,2] = torch.cos(thetax)
    rotx[:,1,2] = -torch.sin(thetax)
    rotx[:,2,1] = torch.sin(thetax)
    roty[:,0,0] = torch.cos(thetay)
    roty[:,2,2] = torch.cos(thetay)
    roty[:,0,2] = torch.sin(thetay)
    roty[:,2,0] = -torch.sin(thetay)
    rotz[:,0,0] = torch.cos(thetaz)
    rotz[:,1,1] = torch.cos(thetaz)
    rotz[:,0,1] = -torch.sin(thetaz)
    rotz[:,1,0] = torch.sin(thetaz)
    rotmat = torch.einsum('bij,bjk->bik', rotz, torch.einsum('bij,bjk->bik', roty, rotx))
    return rotmat

def eulerangles_to_rotvec(thetax, thetay, thetaz):
    rotmat = eulerangles_to_rotmat(thetax, thetay, thetaz)
    return roma.rotmat_to_rotvec(rotmat)

class PosescriptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_image_dir,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        datatype="train",
    ):
        ## load dataset from PoseScript
        # if not os.path.exists(base_image_dir):
        base_image_dir = 'dataset/PoseScript'
        #-- load files from posescript
        # load json file: id to amass file name
        with open(os.path.join(base_image_dir, 'posescript_release', 'ids_2_dataset_sequence_and_frame_index.json')) as f:
            self.ids2name = json.load(f)
        # load id and scripts data
        with open(os.path.join(base_image_dir, 'posescript_release', 'human6293.json')) as f:
            self.ids2scripts_human = json.load(f)

        ids_scripts_automatic = {}
        for name in ['A', 'B', 'C', 'D', 'E', 'F']:
            with open(os.path.join(base_image_dir, 'posescript_release', f'automatic_{name}.json')) as f:
                ids2scripts_A = json.load(f)
                for k, v in ids2scripts_A.items():
                    if k in ids_scripts_automatic:
                        ids_scripts_automatic[k].append(v)
                    else:
                        ids_scripts_automatic[k] = []
                        ids_scripts_automatic[k].append(v)

        self.ids2scripts_automatic = ids_scripts_automatic
        self.datatype = datatype
        # load train/val/test split
        if datatype == 'train':
            with open(os.path.join(base_image_dir, 'posescript_release', 'train_ids.json')) as f:
                self.ids = json.load(f)
        elif datatype == 'val':
            with open(os.path.join(base_image_dir, 'posescript_release', 'val_ids.json')) as f:
                self.ids = json.load(f)
        self.amass_folder = os.path.join(base_image_dir, 'AMASS')
        
        self.short_question_list = TEXT_SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.samples_per_epoch = samples_per_epoch
        self.precision = precision
        self.image_size = image_size
        
    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ## load sample from posescript
        while True:
            idx = random.randint(0, len(self.ids) - 1)
            key = self.ids[idx]     
            key = str(key)   
            # get amass data
            pose_info = self.ids2name[key]
            if pose_info[1].split('/')[0] not in ['BMLhandball', 'MPI_mosh', 'HumanEva']:
                pose_info[1] = pose_info[1].replace('Eyes_Japan_Dataset','EyesJapanDataset')\
                .replace('BioMotionLab_NTroje','BMLrub')\
                .replace('MPI_HDM05','HDM05').replace('MPI_Limits','PosePrior')\
                .replace('MPI_mosh','MoSh').replace('SSM_synced','SSM')\
                .replace('DFaust_67','DFaust').replace('Transitions_mocap','Transitions')
            pose_path = os.path.join(self.amass_folder, pose_info[1])
            if not os.path.exists(pose_path):
                continue
            frame_id = pose_info[2]
            # read pose data
            pose_data = np.load(pose_path, allow_pickle=True)['poses'] # [n_frames, 156]
            smplh_pose = pose_data[int(frame_id)].reshape(-1,3) # [52, 3]
            if smplh_pose.shape[0] != 52:
                continue
                
            break
        # questions and answers
        # 1. get questions from posescript
        pose_script = self.ids2scripts_automatic[key][np.random.randint(0, len(self.ids2scripts_automatic[key]))]
        if key in self.ids2scripts_human:
            pose_script_human = self.ids2scripts_human[key]
            pose_script = random.choice([pose_script, pose_script_human])
        
        # 2. load pose
        smplh_pose = torch.from_numpy(smplh_pose)
        smpl_global_orient = smplh_pose[:1].reshape(-1)
        ## remove z axis
        # rotation matrix to euler angle
        euler_x, euler_y, euler_z = rotvec_to_eulerangles(smpl_global_orient.reshape(-1, 3))
        zeros = torch.zeros_like(euler_z)
        smpl_global_orient = eulerangles_to_rotvec(euler_x, euler_y, zeros).view(-1, 3)
        ## fix global orientation
        default_global_orient = [[1.0000000,  0.0000000,  0.0000000],
                                    [0.0000000,  0.0000000,  -1.0000000],
                                    [0.0000000,  1.0000000, 0.0000000 ]]
        gt_global_orient_correct = torch.tensor(default_global_orient).view(1,3,3).to(smpl_global_orient.device).to(smpl_global_orient.dtype)
        gt_global_orient_rotmat = aa_to_rotmat(smpl_global_orient.reshape(-1, 3)).view(-1, 3, 3)
        gt_global_orient_rotmat = torch.matmul(gt_global_orient_correct, gt_global_orient_rotmat)
        smpl_global_orient = matrix_to_axis_angle(gt_global_orient_rotmat).view(-1)

        smpl_body_pose = smplh_pose[1:22+2].reshape(-1)
        smpl_shape = torch.zeros((10,))
            
        questions = []
        answers = []
        
        ## format questions and answers
        question_template = random.choice(self.short_question_list).format(sent=pose_script)
        questions.append(question_template)
        answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        ### --- placeholder for image to be compatible with hmr/vqa data --
        image = torch.zeros((3, 256, 256))
        image_clip = torch.zeros((3, 336, 336))
        image_path = ''
        resize = image.shape[:2]

        ##         
        return (
            image_path,
            image,
            image_clip,
            conversations,
            resize,
            questions,
            smpl_global_orient,
            smpl_body_pose,
            smpl_shape
        )
        
