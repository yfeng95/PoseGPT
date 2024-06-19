import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import cv2
import numpy as np
from tqdm import tqdm
import os
import pickle
import json

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

## plot bbox on image
def plot_bbox(image, bbox):
    bbox = bbox.astype(np.int)
    # image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0), 2)
    image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,0,255), 2)
    return image

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    # conv = conv_templates[args.conv_mode].copy()
    # if "mpt" in model_name.lower():
    #     roles = ('user', 'assistant')
    # else:
    #     roles = conv.roles

    # from time import time
    # image = load_image(args.image_file)
    # Similar operation in model_worker.py
    # image_tensor = process_images([image], image_processor, args)
    # if type(image_tensor) is list:
    #     image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    # else:
    #     image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    if args.dataset == '4dhumans':
        datafolder = '/fast/yfeng/Projects/GPT/HumanGPT/dataset/humangpt/4dhumans'
    elif args.dataset == 'bedlam':
        datafolder = '/fast/yfeng/Projects/GPT/HumanGPT/dataset/humangpt/bedlam'
    
    # savefolder = os.path.join(datafolder, 'llava')
    # savefolder_highlight = os.path.join(datafolder, 'input_image_highlight')
    savefolder_highlight = os.path.join(datafolder, 'cropped_image')
    savefolder_pure = os.path.join(datafolder, 'input_image')
    os.makedirs(savefolder_highlight, exist_ok=True)
    os.makedirs(savefolder_pure, exist_ok=True)
    savefolder = os.path.join(datafolder, 'llava_json_crop')
    os.makedirs(savefolder, exist_ok=True)
    
    valid_list = []
    question_dict = {
        "pose": "There is a person in the middle of the image, describe what he/she is doing. The answer starts with \"The person\"", # high level human pose
        "pose_detail": "There is a person in the middle of the image, describe the detailed pose of his torso, left and right arms, hands, legs and feets, use short sentences. The answer starts with \"The person\"", # detailed human pose
        "appearance": "Describe how the person in the center of the image look like. The answer starts with \"The person\"", # human appearance
        # "env":"Describe the space relations of the person and the environment. The answer starts with \"The person\"", # background
    }
    
    from glob import glob
    imagepath_list = glob(os.path.join(datafolder, 'cropped_image', '*.png')) 
    imagepath_list = sorted(imagepath_list)
    
    ### split image list into batches
    imagepath_list = imagepath_list[args.batch_idx * args.batch_size: (args.batch_idx + 1) * args.batch_size]
    
    
    question_dict = {
        # 'sys': "You serve as an AI visual analyst for image examination. Your input will be an image containing humans. Your task is to provide descriptions of an individual highlighted by a red rectangle. Your analysis should focus on four attributes: the individual's shape, outfits, behavior and detailed pose. \nFor behavior, if this person is doing specific activities like yoga or sports, provide the detailed name.  For the detailed pose. describe as detail as possible, looking into the torso, left, right arms, hands and legs. \nThe red rectangle represents the bounding box, isolating this single individual for analysis. In cases where multiple individuals appear within the outlined area, your focus should be on the person who occupies the majority of the designated space. \nPlease output the tjree attributes as keys in a JSON file format, each values should be sentences. \n"
        'sys': "You serve as an AI visual analyst for image examination. Your input will be an image containing humans. Your task is to provide descriptions of the individual in the center, if you find multiple persons, only localize one person in the most center, even this person is occluded. Your analysis should focus on four attributes: the individual's shape, outfits, behavior and detailed pose. \nFor behavior, if this person is doing specific activities like yoga or sports, provide the detailed name. \nFor the detailed pose, describe as detail as possible, looking into the torso, left, right arms, hands and legs. Please output these four attributes as keys in a JSON file format, each values should be long sentences. \n"
    }
    for i in tqdm(range(len(imagepath_list))):
        ## load cropped image
        imagepath = imagepath_list[i]
        imagename = imagepath.split('/')[-1].split('.')[0]
        
        
        ## run LLaVA on cropped image
        image = load_image(imagepath)
        image_tensor = process_images([image], image_processor, args)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        outputs_dict = {}
        for key, question in question_dict.items():
            conv = conv_templates[args.conv_mode].copy()
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles

            inp = question
            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                # image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            # import ipdb; ipdb.set_trace()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    # streamer=streamer,
                    use_cache=False) #,
                    # stopping_criteria=[stopping_criteria])
            
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace("</s>", "").replace('\n', '').replace('\\', '')
            # print(outputs)
            # import ipdb; ipdb.set_trace()
            try:
                output = json.loads(outputs)
                # save
                with open(os.path.join(savefolder, imagename + '.json'), 'w') as f:
                    json.dump(output, f, indent=4)
            except:
                print('error')
                        
        #-------- saving results
        # save dict to json
        # with open(os.path.join(savefolder, imagename + '.json'), 'w') as f:
        #     json.dump(outputs_dict, f)
        
        # import ipdb; ipdb.set_trace()
    # save the valid list
    # with open(os.path.join(datafolder, 'valid_list.txt'), 'w') as f:
    #     for item in valid_list:
    #         f.write("%s\n" % item)
    print('done')
    print(f'valid number: {len(valid_list)}')        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="4dhumans")
    parser.add_argument("--dataset", type=str, default="bedlam")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--batch_idx", type=int, default=0)
    # parser.add_argument("--max-example", type=int, default=512)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)


'''
python -m llava.serve.cli_humangpt \
    --model-path liuhaotian/llava-v1.5-13b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
'''