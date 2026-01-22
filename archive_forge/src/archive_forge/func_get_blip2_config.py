import argparse
import requests
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from transformers import (
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
def get_blip2_config(model_name):
    image_size = 364 if 'coco' in model_name else 224
    vision_config = InstructBlipVisionConfig(image_size=image_size).to_dict()
    if 't5-xl' in model_name:
        text_config = T5Config.from_pretrained('google/flan-t5-xl', dense_act_fn='gelu', bos_token_id=1).to_dict()
    elif 't5-xxl' in model_name:
        text_config = T5Config.from_pretrained('google/flan-t5-xxl', dense_act_fn='gelu', bos_token_id=1).to_dict()
    elif 'vicuna-7b' in model_name:
        text_config = LlamaConfig.from_pretrained('decapoda-research/llama-7b-hf', vocab_size=32001).to_dict()
    elif 'vicuna-13b' in model_name:
        text_config = LlamaConfig.from_pretrained('decapoda-research/llama-13b-hf', vocab_size=32001).to_dict()
    else:
        raise ValueError('Model name not supported')
    qformer_config = InstructBlipQFormerConfig(vocab_size=30523).to_dict()
    config = InstructBlipConfig(vision_config=vision_config, text_config=text_config, qformer_config=qformer_config)
    return (config, image_size)