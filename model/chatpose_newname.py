import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
import sys, os
TokenHMR_root = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))), "TokenHMR")
sys.path.insert(0, TokenHMR_root)
from tokenhmr.lib.utils.geometry import aa_to_rotmat, rot6d_to_rotmat
from tokenhmr.lib.models.tokenhmr import TokenHMR
from tokenhmr.lib.configs import get_config
model_cfg = get_config(os.path.join(TokenHMR_root, "data", "checkpoints", "model_config.yaml"))

from .smpl.util import write_obj, tensor2image

class ChatPoseMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(ChatPoseMetaModel, self).__init__(config)

        self.config = config
        self.config.out_dim = kwargs.get("out_dim", 144)
        self.initialize_chatpose_modules(self.config)
        
    def initialize_chatpose_modules(self, config):
        in_dim = config.hidden_size
        out_dim = self.config.out_dim
        # set out_dim same as hmr2 backbone output dim: 1024
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        # init smpl
        from .smpl.SMPLX import SMPLX
        # defin new config
        from yacs.config import CfgNode as CN
        smpl_config = CN()
        smpl_config.smplx_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/SMPLX_NEUTRAL_2020.npz'))
        smpl_config.n_shape = 300
        smpl_config.n_exp = 50
        smpl_config.extra_joint_path = None
        smpl_config.j14_regressor_extra = None
        # smpl_config.dtype = config.torch_dtype
        self.smplx = SMPLX(config=smpl_config) #, dtype=config.torch_dtype)
        # setup renderer
        from .smpl.renderer import set_rasterizer, render_shape
        set_rasterizer(type = 'standard')

class ChatPoseModel(ChatPoseMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(ChatPoseModel, self).__init__(config, **kwargs)

class ChatPoseForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):  
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.vis_steps = kwargs.pop("vis_steps", 10)
        
        super().__init__(config)

        self.model = ChatPoseModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
       
   
    # def forward(self, **kwargs):
    #     if "past_key_values" in kwargs:
    #         return super().forward(**kwargs)
    #     return self.model_forward(**kwargs)

    # def generate(self, **kwargs):
    #     return super().generate(**kwargs)
    
    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        smpl_global_orient: torch.FloatTensor,
        smpl_body_pose: torch.FloatTensor,
        smpl_shape: torch.FloatTensor,
        inference: bool = False,
        **kwargs,
    ):
        batch_size = images_clip.shape[0]
        assert batch_size == len(offset) - 1
        #-------------------------- LLaVA part: image + text -> hidden states --------------------------#
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        ## if there exists IMAGE, then pad 255. 
        image_embeds_len = 575
        if IMAGE_TOKEN_INDEX in input_ids:
            new_seg_token_mask = torch.zeros([seg_token_mask.shape[0], seg_token_mask.shape[1]+image_embeds_len]).bool().cuda()
            for i in range(input_ids.shape[0]):
                ## hack for IMAGE_TOKEN_INDEX if there's image in the front: add image_embeds_len (255 for llava, 575 for llava1.5) zeros in the front
                ## why? if there's image in the front, LLaVA will insert the image embeeding there
                if IMAGE_TOKEN_INDEX in input_ids[i]:
                    new_seg_token_mask[i, image_embeds_len:] = seg_token_mask[i]
                # if no image in the front, pad 255 zeros in the end
                # if no image, the pose token remains the same place
                else:
                    new_seg_token_mask[i, :seg_token_mask.shape[1]] = seg_token_mask[i]
            seg_token_mask = new_seg_token_mask

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        model_output = output
        hidden_states = []
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        
        #### -------------------------- LLaVA part: hidden states -> embeddings -------------------------- ###
        pred_embeddings = last_hidden_state[seg_token_mask]
        
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]
        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        # pred_embeddings = pred_embeddings_
        # text_embeddings = torch.cat(pred_embeddings, dim=0)
        
        # output_ids = model_output.logits.argmax(-1)
        # print('input_ids:', input_ids[0][-10:], 'output_ids: ', output_ids[0][-10:])
        # print('seg_token_counts: ', seg_token_counts, start_i, end_i)
        
        ### -------------------------- SMPL decoder part: embedding -> SMPL parameters and corresponding losses -------------------------- ###
        hmr_loss_list = []
        predictions = None
        ### HMR dataset
        # default_global_orient = [[1.0000000,  0.0000000,  0.0000000],
        #                         [0.0000000,  -1.0000000,  0.0000000],
        #                         [-0.0000000,  0.0000000, -1.0000000 ]]
        # gt_global_orient = torch.tensor(default_global_orient).to(images_clip.device).to(images_clip.dtype).view(1,3,3)
        # gt_global_orient_rotmat = gt_global_orient.expand(batch_size, -1, -1)
        # get gt smpl params
        gt_global_orient_rotmat = aa_to_rotmat(smpl_global_orient.reshape(-1, 3)).view(batch_size, -1, 3, 3).to(images_clip.device).to(images_clip.dtype)
        gt_body_pose_rotmat = aa_to_rotmat(smpl_body_pose.reshape(-1, 3)).view(batch_size, -1, 3, 3).to(images_clip.device).to(images_clip.dtype)
        gt_smpl_shape = smpl_shape.to(images_clip.device).to(images_clip.dtype)
        # gt_smpl_params = {
        #     'global_orient': gt_global_orient_rotmat,
        #     'body_pose': gt_body_pose_rotmat,
        #     'betas': gt_smpl_shape,
        # }
        for i in range(len(pred_embeddings_)):
            text_embeddings = pred_embeddings_[i]
            if text_embeddings.shape[0] == 0:
                continue            
            # hmr_valid = (smpl_body_pose[i:i+1].sum(dim=1) > 0.00001).float()
            
            pred_pose = text_embeddings
            pose_6d = pred_pose.view(-1, 24, 6)
            pose_rotmat = rot6d_to_rotmat(pose_6d).view(-1, 24, 3, 3)
            pred_grot = pose_rotmat[:,0:1]
            pred_body_pose = pose_rotmat[:,1:]
            
            ### compute losses
            hmr_loss = 0
            # parameter loss
            hmr_loss += (pred_body_pose.reshape(1, -1) - gt_body_pose_rotmat.reshape(batch_size, -1)[i:i+1]).abs().mean()
            hmr_loss += (pred_grot.reshape(1, -1) - gt_global_orient_rotmat.reshape(batch_size, -1)[i:i+1]).abs().mean()
            hmr_loss_list.append(hmr_loss)

            ##--- visualize
            if predictions is None and kwargs['global_step'] % self.vis_steps == 0:
                # default camera translation
                # default_cam = [-1.8677e-02,  1.4551e-01,  3.4000e+01]
                default_cam = [-1.8677e-02,  1.4551e-01,  3.000e+01]
                camera_translation = torch.tensor(default_cam).to(images_clip.device).to(images_clip.dtype).view(1,3)
                camera_translation = camera_translation.expand(batch_size, -1) 
                # pose to vertices
                pred_smpl_params = {'global_orient': gt_global_orient_rotmat[i:i+1],
                            'body_pose': pred_body_pose,
                            'betas': gt_smpl_shape[i:i+1]}
                pred_smpl_params['global_orient'] = pred_grot
                ## run smplx to get vertices
                vertices, _, _ = self.smplx(
                        global_pose=pred_smpl_params['global_orient'],
                        body_pose=pred_smpl_params['body_pose'])
                vertices = vertices.detach()
                gt_smpl_params = {'global_orient': gt_global_orient_rotmat[i:i+1],
                            'body_pose': gt_body_pose_rotmat[i:i+1],
                            }
                gt_vertices, _, _ = self.smplx(
                    global_pose=gt_smpl_params['global_orient'],
                    body_pose=gt_smpl_params['body_pose'])
                faces = self.smplx.faces_tensor
                ## render image
                from .smpl.renderer import set_rasterizer, render_shape
                set_rasterizer(type = 'standard')
                shape_image = render_shape(vertices.float(), faces.unsqueeze(), image_size=512)
                ## save image
                import cv2
                cv2.imwrite('test.png', tensor2image(shape_image[0]))

                print(faces.shape)
                import ipdb; ipdb.set_trace()
                
                ## save obj file 
                write_obj(
                        'test.obj', vertices.cpu().numpy(), faces,
                        colors=colors,
                        texture=uvmap,
                        uvcoords=uvcoords,
                        uvfaces=uvfaces,
                        inverse_face_order=False,
                        normal_map=opdict.get('normal_map'),
                    )

            ### hmr data
            # else: 
            #     # extract image embedding from hmr backbone
            #     # conditioning_feats = self.model.hmr.backbone(images[:,:,:,32:-32])
            #     # conditioning_feats = self.model.hmr.backbone(hmr_batch['img'][:,:,:,32:-32])
                
            #     # combine with LLaVA embeddings
            #     # conditioning_feats = text_embeddings[:,:,None,None]
            #     if not self.keep_hmr_embedding:
            #         hmr_batch['text_embeddings'] = text_embeddings
            #         if self.text_embeddings_for_global:
            #             hmr_batch['text_embeddings_for_global'] = True
            #     hmr_output = self.model.hmr.forward_step(hmr_batch) #, conditioning_feats=conditioning_feats, text_embeddings=text_embeddings)
                
            #     ## compute losses 
            #     hmr_loss = self.model.hmr.compute_loss(hmr_batch, hmr_output)
            #     ## visualize
            #     predictions = self.model.hmr.tensorboard_logging(hmr_batch, hmr_output, step_count=0, 
            #                                                     write_to_summary_writer=False,
            #                                                     )
                # save predictions to images
                # if self.forward_count % 10 == 0:
                #     cv2.imwrite(f'/fast/yfeng/Projects/TokenHMR/HumanGPT/runs/test_hmr/{self.forward_count:05}.png', (np.transpose(predictions.cpu().numpy(), (1,2,0))[:,:,[2,1,0]]*255).astype(np.uint8))
                # self.forward_count += 1
        # else:
        #     hmr_loss = 0
        #     predictions = None #torch.zeros((batch_size, 3, 224, 224)).to(images_clip.device).to(images_clip.dtype)
        hmr_loss = 0
        for hmr_loss_i in hmr_loss_list:
            hmr_loss += hmr_loss_i
        hmr_loss = hmr_loss/(len(hmr_loss_list)+1e-8)

        if inference:
            return {
                "pred_embeddings": pred_embeddings,
            }

        output = model_output.logits

        loss = 0.
        self.ce_loss_weight = 1.0
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        loss += ce_loss

        self.hmr_loss_weight = 0.1
        hmr_loss = hmr_loss * self.hmr_loss_weight
        loss += hmr_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "hmr_loss": hmr_loss,
            "predictions": predictions,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        smpl_global_orient=None,
        smpl_body_pose=None,
        smpl_shape=None,
        max_new_tokens=32,
        tokenizer=None,
        **kwargs,
    ):
        with torch.no_grad():
            # import ipdb; ipdb.set_trace()
            # output = super().forward(
            #     images=images_clip,
            #     input_ids=input_ids,
            #     output_hidden_states=True,
            # )
            # outputs = self.generate(
            #     images=images_clip,
            #     input_ids=input_ids,
            #     max_new_tokens=max_new_tokens,
            #     num_beams=1,
            #     output_hidden_states=True,
            #     return_dict_in_generate=True,
            # )
            # import ipdb; ipdb.set_trace()
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1][-1]
            # import ipdb; ipdb.set_trace()
            output_ids = outputs.sequences
            # del outputs
        
            # import ipdb; ipdb.set_trace()
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            seg_token_mask[:,:input_ids.shape[1]] = False
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            # if 'img' in hmr_batch.keys() and hmr_batch['smpl_params']['betas'].sum() == 0:
            #     seg_token_mask = seg_token_mask
            # else:
            ## if there exists IMAGE, then pad 255. 
            image_embeds_len = 575
            if IMAGE_TOKEN_INDEX in input_ids:
                new_seg_token_mask = torch.zeros([seg_token_mask.shape[0], seg_token_mask.shape[1]+image_embeds_len]).bool().cuda()
                for i in range(input_ids.shape[0]):
                    ## hack for IMAGE_TOKEN_INDEX if there's image in the front: add 255 zeros in the front
                    ## why? if there's image in the front, LLaVA will insert the image embeeding there
                    if IMAGE_TOKEN_INDEX in input_ids[i]:
                        new_seg_token_mask[i, image_embeds_len:] = seg_token_mask[i]
                    # if no image in the front, pad 255 zeros in the end
                    # if no image, the pose token remains the same place
                    else:
                        new_seg_token_mask[i, :seg_token_mask.shape[1]] = seg_token_mask[i]
                seg_token_mask = new_seg_token_mask

            # # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            # seg_token_mask = torch.cat(
            #     [
            #         torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
            #         seg_token_mask,
            #     ],
            #     dim=1,
            # )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_
            
            ### -------------------------- SMPL decoder part: embedding -> SMPL parameters and corresponding losses -------------------------- ###
            text_embeddings = torch.cat(pred_embeddings, dim=0)
            text_embeddings = text_embeddings[-1:]
            
            hmr_loss_list = []
            predictions = None
            pred_smpl_params = None
            ### HMR dataset
            batch_size = images_clip.shape[0]
            # default_global_orient = [[1.0000000,  0.0000000,  0.0000000],
            #                         [0.0000000,  -1.0000000,  0.0000000],
            #                         [-0.0000000,  0.0000000, -1.0000000 ]]
            # default_global_orient = [[1.0000000,  0.0000000,  0.0000000],
            #                         [0.0000000,  0.0000000,  -1.0000000],
            #                         [0.0000000,  1.0000000, 0.0000000 ]]
            # gt_global_orient_correct = torch.tensor(default_global_orient).to(images_clip.device).to(images_clip.dtype).view(1,3,3)
            # gt_global_orient_rotmat_correct = gt_global_orient_correct.expand(batch_size, 1, -1, -1)
            
            
            # gt_global_orient_rotmat = torch.matmul(gt_global_orient_rotmat, gt_global_orient_rotmat_correct)
            # gt_global_orient_rotmat = torch.matmul(gt_global_orient_rotmat_correct, gt_global_orient_rotmat)
            # gt_smpl_params = {
            #     'global_orient': gt_global_orient_rotmat,
            #     'body_pose': gt_body_pose_rotmat,
            #     'betas': gt_smpl_shape,
            # }
            # print('print_global_orient: ', self.predict_global_orient)
            for i in range(1):
                # text_embeddings = pred_embeddings_[i]
                if text_embeddings.shape[0] == 0:
                    continue            
                # hmr_valid = (smpl_body_pose[i:i+1].sum(dim=1) > 0.00001).float()
                
                # get pred smpl params
                # if self.config.out_dim == 1024:
                #     smpl_thetas6D, cls_logits_softmax = self.model.hmr.smpl_head.decpose(x=None, text_embeddings=text_embeddings)        
                #     pred_bpose = rot6d_to_rotmat(smpl_thetas6D.reshape(-1, 21, 6)).reshape(-1, 21*9).view(1, -1, 3, 3) # [bs, 21*9]
                #     pred_body_pose = torch.cat([pred_bpose, gt_body_pose_rotmat[i:i+1,21:]], dim=1) # [bs, 23, 3, 3]
                #     ## global
                #     if self.predict_global_orient:
                #         pred_grot = self.model.hmr.smpl_head.decpose_grot(text_embeddings)
                #         # 6d to matrix
                #         pred_grot = rot6d_to_rotmat(pred_grot.reshape(-1, 1, 6)).reshape(-1, 3, 3)       
                #         # pred_grot = torch.bmm(pred_grot.view(-1,3,3), gt_global_orient_rotmat_correct.view(-1,3,3)).view(batch_size, -1, 3, 3)         
                #     else:
                #         pred_grot = gt_global_orient_rotmat[i:i+1]
                # else:
                pred_pose = text_embeddings
                pose_6d = pred_pose.view(-1, 24, 6)
                pose_rotmat = rot6d_to_rotmat(pose_6d).view(-1, 24, 3, 3)
                pred_grot = pose_rotmat[:,0:1]
                pred_body_pose = pose_rotmat[:,1:]
            
                # if self.predict_global_orient:
                #     pred_grot = pred_grot
                # else:
                #     if smpl_global_orient is not None:
                #         gt_global_orient_rotmat = aa_to_rotmat(smpl_global_orient.reshape(-1, 3)).view(batch_size, -1, 3, 3).to(images_clip.device).to(images_clip.dtype)
                #         pred_grot = gt_global_orient_rotmat[i:i+1]
                #     else:
                #         pred_grot = pred_grot
                    
                ##--- visualize
                if predictions is None:
                    # default camera translation
                    default_cam = [-1.8677e-02,  1.4551e-01,  3.0000e+01]
                    camera_translation = torch.tensor(default_cam).to(images_clip.device).to(images_clip.dtype).view(1,3)
                    camera_translation = camera_translation.expand(1, -1) 
                    # pose to vertices
                    vertices = self.model.smplx(global_pose=pred_grot, body_pose=pred_body_pose[:,:21])[0]

                    if smpl_global_orient is not None:
                        # get gt smpl params
                        smpl_global_orient_curr = smpl_global_orient.clone()
                        gt_global_orient_rotmat = aa_to_rotmat(smpl_global_orient_curr.reshape(-1, 3)).view(batch_size, -1, 3, 3).to(images_clip.device).to(images_clip.dtype)
                        gt_body_pose_rotmat = aa_to_rotmat(smpl_body_pose.reshape(-1, 3)).view(batch_size, -1, 3, 3).to(images_clip.device).to(images_clip.dtype)
                        gt_vertices = self.model.smplx(global_pose=gt_global_orient_rotmat[i:i+1], body_pose=gt_body_pose_rotmat[i:i+1,:21])[0]
                    else:
                        gt_vertices = vertices.clone()
                    
                    faces = self.model.smplx.faces_tensor

                    ## render image
                    from .smpl.renderer import set_rasterizer, render_shape
                    set_rasterizer(type = 'standard')
                    shape_image = render_shape(vertices.float(), faces.unsqueeze(dim=0), image_size=512)
                    ## save image
                    import cv2
                    cv2.imwrite('test.png', tensor2image(shape_image[0]))
                    ## save obj file
                    write_obj(
                        'test.obj', vertices.float().cpu().numpy().squeeze(), faces.cpu().numpy(),
                    )
                    exit()
                    ## render image


                    vis_images = images[i:i+1] * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
                    vis_images = vis_images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
                    vis_images_clip = images_clip[i:i+1] * torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).reshape(1,3,1,1)
                    vis_images_clip = vis_images_clip + torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).reshape(1,3,1,1)
                    #  resize clip to vis image size
                    vis_images_clip = F.interpolate(vis_images_clip, size=vis_images.shape[-1], mode='bilinear', align_corners=False)
                    predictions = self.model.hmr.mesh_renderer.visualize_vertices(
                        vertices.float().cpu().numpy(), 
                        camera_translation=camera_translation.float().cpu().numpy(), 
                        images=vis_images.float().cpu().numpy(), 
                        gt_vertices=gt_vertices.float().cpu().numpy(), 
                        clip_images=vis_images_clip.float().cpu().numpy(),
                    )
        if kwargs.get('return_smpl', False):
            return output_ids, predictions, pred_smpl_params
        
        return output_ids, predictions
