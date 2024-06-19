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

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST

'''
Load dataset for HMR task
1. get dataset from 4DHumans
2. get sample from dataset, and process it into a format that can be used by LLaVA
'''
DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])
DEFAULT_IMG_SIZE = 256
class HMRDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
    ):
        ## load dataset from 4DHumans
        import sys, os
        TokenHMR_root = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))), "..", "model", "TokenHMR")
        sys.path.insert(0, os.path.join(TokenHMR_root, "4DHumans"))
        from hmr2.configs import dataset_config, default_config
        from hmr2.datasets import HMR2DataModule
        from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
        import yaml
        from omegaconf import DictConfig, OmegaConf
        # model_cfg_path = '/is/cluster/fast/scratch/sdwivedi/TokenHMR/models/model_config.yaml'
        model_cfg_path = '/yfeng/Projects/GPT/HumanGPT/model/TokenHMR/logs/1023_BD_vTok/runs/tokenhmr_yao/hmr2_model_config.yaml'
        cfg = default_config()
        cfg.merge_from_file(os.path.join(TokenHMR_root, "4DHumans", "hmr2", "configs_hydra", "experiment", "default.yaml"))
        cfg.merge_from_file(model_cfg_path)
        # cfg.DATASETS.DATASET_DIR = os.path.join(TokenHMR_root, cfg.DATASETS.DATASET_DIR)
        cfg.DATASETS.DATASET_DIR = os.path.join(TokenHMR_root, '4DHumans/hmr2_datasets/hmr2_training_data')
        # data_cfg_path = os.path.join(TokenHMR_root, "4DHumans", "hmr2_data","hmr2_models", "dataset_config.yaml")
        # data_cfg_path = '/fast/yfeng/Projects/GPT/HumanGPT/model/TokenHMR/logs/1023_BD_vTok/runs/bedlam_3dpw/dataset_config.yaml'
        data_cfg_path = '/fast/yfeng/Projects/GPT/HumanGPT/model/TokenHMR/logs/1023_BD_vTok/runs/tokenhmr_yao/dataset_config.yaml'
        dataset_cfg = yaml.load(open(data_cfg_path, "r"), Loader=yaml.FullLoader) 
        dataset_cfg = OmegaConf.create(dataset_cfg)
        datamodule = HMR2DataModule(cfg, dataset_cfg)
        # datamodule.setup(only_train=True)
        datamodule.setup()
        self.hmr_dataset = datamodule.train_dataset
        self.hmr_dataset_iter = iter(self.hmr_dataset)
        # from hmr2.datasets.synthetic_dataset import BedlamDataset
        # self.hmr_dataset = BedlamDataset(cfg, dataset_cfg['BEDLAM'], 'orbit-archviz-10')
        # import ipdb; ipdb.set_trace()
        # self.hmr_dataloader = torch.utils.data.DataLoader(
        #     self.hmr_dataset, 
        #     1, 
        #     drop_last=True, 
        #     shuffle=True,
        #     num_workers=1)
        # import ipdb; ipdb.set_trace()
        # # test image
        # batch = next(iter(self.hmr_dataset))
        # img = batch['img']
        # img = img * DEFAULT_STD[:, None, None] + DEFAULT_MEAN[:, None, None]
        # cv2.imwrite('test.png', img.transpose(1,2,0)[..., ::-1].astype(np.uint8))
        
        ## --- remaining code is from Lisa's code ---
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        
    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ## load sample from 4DHumans
        try:
            batch = next(self.hmr_dataset_iter)
        except:
            self.hmr_dataset_iter = iter(self.hmr_dataset)
            batch = next(self.hmr_dataset_iter)
        # if idx % 2 == 0:
        #     batch = self.hmr_dataset[1]
        # else:
        # batch = self.hmr_dataset[2000]
        # idx = random.randint(0, 2)
        # idx_list = [20000, 100, 1000]
        # idx = idx_list[idx]
        # idx = 20000
        # idx = random.randint(0, len(self.hmr_dataset) - 1)
        # batch = self.hmr_dataset[idx]
        hmr_img = batch['img']
        # recover image color according to the mean and std
        img = hmr_img * DEFAULT_STD[:, None, None] + DEFAULT_MEAN[:, None, None]
        image = cv2.cvtColor(img.transpose(1,2,0)[..., ::-1].astype(np.uint8), cv2.COLOR_BGR2RGB)
        ## --- remaining code is from Lisa to process the sample ---        
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        resize = image_clip.shape[:2]
        # image = hmr_img  # preprocess image for HMR2
        smpl_params = batch['smpl_params']
        # 1. full body pose
        smpl_body_pose = smpl_params['body_pose']
        smpl_body_pose = torch.from_numpy(smpl_body_pose).float() # (69,)
        # 2. global orientation
        smpl_global_orient = smpl_params['global_orient'] # (3,)
        smpl_global_orient = torch.from_numpy(smpl_global_orient).float()
        # 3. shape
        smpl_shape = smpl_params['betas'] # (10,)
        smpl_shape = torch.from_numpy(smpl_shape).float()
        ### keypoint
        # 1. 2D keypoint
        smpl_keypoints_2d = batch['keypoints_2d']
        smpl_keypoints_2d = torch.from_numpy(smpl_keypoints_2d).float()
        # 2. 3D keypoint
        smpl_keypoints_3d = batch['keypoints_3d']
        smpl_keypoints_3d = torch.from_numpy(smpl_keypoints_3d).float()
        
        # questions and answers
        questions = []
        answers = []
        for i in range(1):
            question_template = random.choice(self.short_question_list)
            # question_template = self.short_question_list[i]
            questions.append(question_template)

            answers.append(random.choice(self.answer_list))
            # answers.append(self.answer_list[i])

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = torch.from_numpy(batch['img'])
        image_path = ''
    
        # 
        return (
            image_path,
            image,
            image_clip,
            conversations,
            resize,
            questions,
            smpl_global_orient,
            smpl_body_pose,
            smpl_shape,
            # smpl_keypoints_2d,
            # smpl_keypoints_3d
        )


class BEDLAMDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
    ):
        ## load dataset from 4DHumans
        import sys, os
        TokenHMR_root = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))), "..", "model", "TokenHMR")
        sys.path.insert(0, os.path.join(TokenHMR_root, "4DHumans"))
        from hmr2.configs import dataset_config, default_config
        from hmr2.datasets import HMR2DataModule
        from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
        import yaml
        from omegaconf import DictConfig, OmegaConf
        # model_cfg_path = '/is/cluster/fast/scratch/sdwivedi/TokenHMR/models/model_config.yaml'
        model_cfg_path = '/fast/yfeng/Projects/GPT/HumanGPT/model/TokenHMR/logs/1023_BD_vTok/runs/tokenhmr_yao/bedlam_model_config.yaml'
        cfg = default_config()
        cfg.merge_from_file(os.path.join(TokenHMR_root, "4DHumans", "hmr2", "configs_hydra", "experiment", "default.yaml"))
        cfg.merge_from_file(model_cfg_path)
        # cfg.DATASETS.DATASET_DIR = os.path.join(TokenHMR_root, cfg.DATASETS.DATASET_DIR)
        cfg.DATASETS.DATASET_DIR = os.path.join(TokenHMR_root, '4DHumans/hmr2_datasets/hmr2_training_data')
        # data_cfg_path = os.path.join(TokenHMR_root, "4DHumans", "hmr2_data","hmr2_models", "dataset_config.yaml")
        # data_cfg_path = '/fast/yfeng/Projects/GPT/HumanGPT/model/TokenHMR/logs/1023_BD_vTok/runs/bedlam_3dpw/dataset_config.yaml'
        data_cfg_path = '/fast/yfeng/Projects/GPT/HumanGPT/model/TokenHMR/logs/1023_BD_vTok/runs/tokenhmr_yao/dataset_config.yaml'
        dataset_cfg = yaml.load(open(data_cfg_path, "r"), Loader=yaml.FullLoader) 
        dataset_cfg = OmegaConf.create(dataset_cfg)
        datamodule = HMR2DataModule(cfg, dataset_cfg)
        # datamodule.setup(only_train=True)
        datamodule.setup()
        self.hmr_dataset = datamodule.train_dataset
        # from hmr2.datasets.synthetic_dataset import BedlamDataset
        # self.hmr_dataset = BedlamDataset(cfg, dataset_cfg['BEDLAM'], 'orbit-archviz-10')
        # import ipdb; ipdb.set_trace()
        # self.hmr_dataloader = torch.utils.data.DataLoader(
        #     self.hmr_dataset, 
        #     1, 
        #     drop_last=True, 
        #     shuffle=True,
        #     num_workers=1)
        # import ipdb; ipdb.set_trace()
        # # test image
        # batch = next(iter(self.hmr_dataset))
        # img = batch['img']
        # img = img * DEFAULT_STD[:, None, None] + DEFAULT_MEAN[:, None, None]
        # cv2.imwrite('test.png', img.transpose(1,2,0)[..., ::-1].astype(np.uint8))
        
        ## --- remaining code is from Lisa's code ---
        self.samples_per_epoch = samples_per_epoch

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        
    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ## load sample from BEDLAM
        idx = random.randint(0, len(self.hmr_dataset) - 1)
        batch = self.hmr_dataset[idx]
        hmr_img = batch['img']
        # recover image color according to the mean and std
        img = hmr_img * DEFAULT_STD[:, None, None] + DEFAULT_MEAN[:, None, None]
        image = cv2.cvtColor(img.transpose(1,2,0)[..., ::-1].astype(np.uint8), cv2.COLOR_BGR2RGB)
        ## --- remaining code is from Lisa to process the sample ---        
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        # image = hmr_img  # preprocess image for HMR2
        resize = image.shape[:2]
        # process labels for hmr
        # TODO, ref: https://github.com/saidwivedi/TokenHMR/blob/d1703d6d43b78438454d30627eb07d6e385ecb95/4DHumans/hmr2/models/hmr2.py#L229
        ### load SMPL parameters
        smpl_params = batch['smpl_params']
        # 1. full body pose
        smpl_body_pose = smpl_params['body_pose']
        smpl_body_pose = torch.from_numpy(smpl_body_pose).float() # (69,)
        # 2. global orientation
        smpl_global_orient = smpl_params['global_orient'] # (3,)
        smpl_global_orient = torch.from_numpy(smpl_global_orient).float()
        # 3. shape
        smpl_shape = smpl_params['betas'] # (10,)
        smpl_shape = torch.from_numpy(smpl_shape).float()
        # questions and answers
        questions = []
        answers = []
        for i in range(1):
            question_template = random.choice(self.short_question_list)
            # question_template = self.short_question_list[i]
            questions.append(question_template)

            answers.append(random.choice(self.answer_list))
            # answers.append(self.answer_list[i])

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        # image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        image = torch.from_numpy(batch['img'])
        image_path = ''
    
        # 
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
