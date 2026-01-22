import argparse
import json
import os
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ConvNextImageProcessor, ConvNextV2Config, ConvNextV2ForImageClassification
from transformers.image_utils import PILImageResampling
from transformers.utils import logging
def convert_preprocessor(checkpoint_url):
    if '224' in checkpoint_url:
        size = 224
        crop_pct = 224 / 256
    elif '384' in checkpoint_url:
        size = 384
        crop_pct = None
    else:
        size = 512
        crop_pct = None
    return ConvNextImageProcessor(size=size, crop_pct=crop_pct, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], resample=PILImageResampling.BICUBIC)