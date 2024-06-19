from enum import Enum

import numpy as np
import torch
import torch.distributed as dist

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# HMR_SHORT_QUESTION_LIST = [
#     "Can you give the SMPL pose of this person?",
#     "Please output this person's SMPL pose.",
#     "Describe what this perosn is doing using SMPL pose.",
#     "What's the SMPL pose of this person?",
#     "Use SMPL to describe this person's pose."
# ]

HMR_SHORT_QUESTION_LIST = [
    "I have a description of a person's pose, can you give the SMPL pose of this person?",
    "Give you a word descrption of a human, please output the SMPL pose.",
    "Describe what this perosn is doing using SMPL pose.",
    "What's the SMPL pose of this person?",
    "Use SMPL pose to describe this person's behavior."
]

# TEXT_SHORT_QUESTION_LIST = [
#     "Can you give the SMPL pose?",
#     "Please output this person's SMPL pose.",
#     "Give the SMPL pose.",
#     "What's the SMPL pose of it?",
#     "Use SMPL to describe the pose."
# ]
TEXT_SHORT_QUESTION_LIST = [
    "I have a word description of a person's pose, can you give the SMPL pose of this person? {sent}",
    "There is a person: {sent} Please output this person's SMPL pose.",
    "{sent} Give the SMPL pose.",
    "What's the SMPL pose of this person? {sent}",
    "Use SMPL pose to describe this person's behavior. {sent}",
    "There is a person doing this: {sent} Can you use SMPL pose to describe the pose?",
    "A person is described as: {sent} Use the SMPL pose to reflect this.",
    "Human pose is described as words: {sent} The SMPL pose is?",
    "Human pose can be described as words: {sent} And it can also be described as SMPL pose format, can you output this?",
]


SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you predict the SMPL pose of the person in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "There is a person in the middle of the image, please output this person's SMPL pose.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is the human pose in this image? Please respond with SMPL pose.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is the person doing in this image? Please output SMPL pose.",
    DEFAULT_IMAGE_TOKEN + "\n" + "There is a person in the middle of the image, use SMPL to describe the pose.",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with SMPL pose.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output SMPL pose.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output SMPL pose and explain the pose.",
    "Please output SMPL pose and explain the reason.",
    "Please output SMPL pose and give some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the SMPL pose is [SEG].",
    "[SEG].",
    "The SMPL pose is [SEG].",
    "The SMPL pose of the person is [SEG].",
    "The SMPL format of this person's pose is [SEG].",
]
