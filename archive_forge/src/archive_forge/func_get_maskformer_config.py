import argparse
import json
import pickle
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, MaskFormerImageProcessor, ResNetConfig
from transformers.utils import logging
def get_maskformer_config(model_name: str):
    if 'resnet101c' in model_name:
        raise NotImplementedError('To do')
    elif 'resnet101' in model_name:
        backbone_config = ResNetConfig.from_pretrained('microsoft/resnet-101', out_features=['stage1', 'stage2', 'stage3', 'stage4'])
    else:
        backbone_config = ResNetConfig.from_pretrained('microsoft/resnet-50', out_features=['stage1', 'stage2', 'stage3', 'stage4'])
    config = MaskFormerConfig(backbone_config=backbone_config)
    repo_id = 'huggingface/label-files'
    if 'ade20k-full' in model_name:
        config.num_labels = 847
        filename = 'maskformer-ade20k-full-id2label.json'
    elif 'ade' in model_name:
        config.num_labels = 150
        filename = 'ade20k-id2label.json'
    elif 'coco-stuff' in model_name:
        config.num_labels = 171
        filename = 'maskformer-coco-stuff-id2label.json'
    elif 'coco' in model_name:
        config.num_labels = 133
        filename = 'coco-panoptic-id2label.json'
    elif 'cityscapes' in model_name:
        config.num_labels = 19
        filename = 'cityscapes-id2label.json'
    elif 'vistas' in model_name:
        config.num_labels = 65
        filename = 'mapillary-vistas-id2label.json'
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    return config