import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig

from model.chatpose import ChatPoseForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

import json
from tqdm import tqdm
from io import BytesIO
from transformers import TextStreamer

def parse_args(args):
    parser = argparse.ArgumentParser(description="ChatPose chat")
    # parser.add_argument("--exp_name", default='CHATPOSE-V0', type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--version", default="YaoFeng/CHATPOSE-V0")
    # parser.add_argument("--image_file", type=str, default="dataset/baber.png")
    parser.add_argument("--image_file", type=str, default=None)
    parser.add_argument("--image_dir", default="./dataset/Yoga-82")
    parser.add_argument("--json_path", default="./dataset/Yoga-82/yoga_dataset.json")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--out_dim", default=144, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument(
        "--dataset", default="hmr||vqa", type=str
    )
    parser.add_argument("--text_embeddings_for_global", action="store_true", default=False)
    parser.add_argument("--predict_global_orient", action="store_true", default=False)
    parser.add_argument("--cat_image_embeds", action="store_true", default=False)
    
    return parser.parse_args(args)

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def visualize_LLM(save_path, questions, answers, visualizations):
    ''' Visualize the questions, answers and visualizations (of SMPL pose) in a single image
    '''
    import matplotlib.pyplot as plt

    # Define the question and answers
    questions = [f'\nQ{i}: {question}' for i, question in enumerate(questions)]
    answers = [f'\nA{i}: {answer.replace("[SEG] ", "[POSE]").replace("[SEG]", "[POSE]")}' for i, answer in enumerate(answers)]

    # Set the maximum text width (adjust as needed)
    if len(questions) == 1:
        questions.append('')
        answers.append('')
        visualizations.append(None)
    # Create a figure and axis
    fig, ax = plt.subplots(nrows=len(questions), ncols=1, figsize=(12, 12))

    # Add the question and answers as wrapped text below the image
    # add one subplot per question
    for k, ax_ in enumerate(ax.flat):
        if visualizations[k] is not None:
            image = (np.transpose(visualizations[k].cpu().numpy(), (1,2,0))*255).astype(np.uint8)
            ax_.imshow(image)
        ax_.axis('off')  # Hide axis
        # add title in the bottom
        ax_.set_title(questions[k] + '\n' + answers[k], wrap=True, color='g')

    # Save the visualization as a PNG
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
def main(args):
    args = parse_args(args)
    if args.exp_name is not None:
        # save name is caplitalized version of exp_name
        save_name = args.exp_name.upper()
        args.version = f"./checkpoints/{save_name}"
        args.vis_save_path = f"./vis_output/{save_name}"
    else:
        save_name = args.version.split('/')[-1]
        args.vis_save_path = f"./vis_output/{save_name}"
    os.makedirs(args.vis_save_path, exist_ok=True)
    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    kwargs.update({"out_dim": args.out_dim})
    kwargs.update({"text_embeddings_for_global": args.text_embeddings_for_global})
    kwargs.update({"predict_global_orient": args.predict_global_orient})
    kwargs.update({"cat_image_embeds": args.cat_image_embeds})
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )
    model = ChatPoseForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    
    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = vision_tower.image_processor
    
    model.eval()

    image_path = args.image_file
    if not os.path.exists(image_path):
        print("Image File not found in {}, use ChatPose without image input".format(image_path))
        image_clip = torch.zeros(1, 3, 336, 336).cuda()
        image = torch.zeros(1, 3, 256, 256).cuda()
        # image_clip = image_clip.unsqueeze(0).cuda()
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()
        pad_image = False
        image_path = 'textonly.png'
    else:
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]
        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        # image_clip = image_clip.unsqueeze(0).cuda()
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()
        image = image_clip.clone()
        image = F.interpolate(image.float(), size=[256,256], mode='bilinear', align_corners=False).to(image_clip.dtype)
        pad_image = True
        
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    roles = conv.roles
    
    questions = []
    answers = []
    visualizations = []

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        questions.append(inp)
        if pad_image:
            prompt = inp#input("Please input your prompt: ")
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            if args.use_mm_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                conv.append_message(conv.roles[0], prompt)
                pad_image = False
        else:
            prompt = inp
            conv.append_message(conv.roles[0], prompt)
            
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
           
    
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        output_ids, predictions, pred_smpl_params = model.evaluate(
            image_clip,
            image,
            input_ids,
            max_new_tokens=512,
            tokenizer=tokenizer,
            return_smpl=True,
        )
        output_ids = output_ids[0, input_ids.shape[1]:]
        text_output = tokenizer.decode(output_ids, skip_special_tokens=False).strip().replace("</s>", "")
        text_output = text_output.replace("[SEG] ", "[POSE]").replace("[SEG]", "[POSE]") # old version uses special [SEG], use real [POSE] should be the same
        print(text_output)
        conv.messages[-1][-1] = text_output
        answers.append(text_output)
        visualizations.append(predictions)
        
    imagename = os.path.basename(image_path)
    save_path = os.path.join(args.vis_save_path, imagename)
    visualize_LLM(save_path, questions, answers, visualizations)        
    print(f"Visualization saved as {save_path}")
    if pred_smpl_params is not None:
        with open(save_path.replace('.png', '.pkl').replace('.jpg', '.pkl'), 'wb') as f:
            import pickle
            pickle.dump(pred_smpl_params, f)
        ## save obj file
        from model.smpl.util import write_obj
        objpath = save_path.replace('.png', '.obj').replace('.jpg', '.obj')
        write_obj(objpath, 
                pred_smpl_params['vertices'].float().cpu().numpy().squeeze(), 
                pred_smpl_params['faces'].cpu().numpy(),
                )
        print(f"SMPL parameters saved as {save_path.replace('.png', '.pkl').replace('.jpg', '.pkl')}")
        print(f"SMPL mesh saved as {objpath}")
        
if __name__ == "__main__":
    main(sys.argv[1:])         
