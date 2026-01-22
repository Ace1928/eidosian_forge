import argparse
import json
from pathlib import Path
import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import DeiTConfig, DeiTForImageClassificationWithTeacher, DeiTImageProcessor
from transformers.utils import logging
@torch.no_grad()
def convert_deit_checkpoint(deit_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our DeiT structure.
    """
    config = DeiTConfig()
    base_model = False
    config.num_labels = 1000
    repo_id = 'huggingface/label-files'
    filename = 'imagenet-1k-id2label.json'
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    config.patch_size = int(deit_name[-6:-4])
    config.image_size = int(deit_name[-3:])
    if deit_name[9:].startswith('tiny'):
        config.hidden_size = 192
        config.intermediate_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 3
    elif deit_name[9:].startswith('small'):
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 6
    if deit_name[9:].startswith('base'):
        pass
    elif deit_name[4:].startswith('large'):
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
    timm_model = timm.create_model(deit_name, pretrained=True)
    timm_model.eval()
    state_dict = timm_model.state_dict()
    rename_keys = create_rename_keys(config, base_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model)
    model = DeiTForImageClassificationWithTeacher(config).eval()
    model.load_state_dict(state_dict)
    size = int(256 / 224 * config.image_size)
    image_processor = DeiTImageProcessor(size=size, crop_size=config.image_size)
    encoding = image_processor(images=prepare_img(), return_tensors='pt')
    pixel_values = encoding['pixel_values']
    outputs = model(pixel_values)
    timm_logits = timm_model(pixel_values)
    assert timm_logits.shape == outputs.logits.shape
    assert torch.allclose(timm_logits, outputs.logits, atol=0.001)
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model {deit_name} to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)