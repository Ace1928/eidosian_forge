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
def get_processor():
    image_processor = EfficientNetImageProcessor(do_center_crop=True, rescale_factor=1 / 127.5, rescale_offset=True, do_normalize=False, include_top=False, resample=Image.BILINEAR)
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
    tokenizer.model_max_length = 64
    processor = AlignProcessor(image_processor=image_processor, tokenizer=tokenizer)
    return processor