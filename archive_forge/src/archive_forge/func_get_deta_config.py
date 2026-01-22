import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import cached_download, hf_hub_download, hf_hub_url
from PIL import Image
from transformers import DetaConfig, DetaForObjectDetection, DetaImageProcessor
from transformers.utils import logging
def get_deta_config():
    config = DetaConfig(num_queries=900, encoder_ffn_dim=2048, decoder_ffn_dim=2048, num_feature_levels=5, assign_first_stage=True, with_box_refine=True, two_stage=True)
    config.num_labels = 91
    repo_id = 'huggingface/label-files'
    filename = 'coco-detection-id2label.json'
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type='dataset')), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    return config