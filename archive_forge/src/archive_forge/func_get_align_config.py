import argparse
import os
import align
import numpy as np
import requests
import tensorflow as tf
import torch
from PIL import Image
from tokenizer import Tokenizer
from transformers import (
from transformers.utils import logging
def get_align_config():
    vision_config = EfficientNetConfig.from_pretrained('google/efficientnet-b7')
    vision_config.image_size = 289
    vision_config.hidden_dim = 640
    vision_config.id2label = {'0': 'LABEL_0', '1': 'LABEL_1'}
    vision_config.label2id = {'LABEL_0': 0, 'LABEL_1': 1}
    vision_config.depthwise_padding = []
    text_config = BertConfig()
    config = AlignConfig.from_text_vision_configs(text_config=text_config, vision_config=vision_config, projection_dim=640)
    return config