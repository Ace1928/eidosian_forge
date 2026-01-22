import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging
@torch.no_grad()
def convert_swiftformer_checkpoint(swiftformer_name, pytorch_dump_folder_path, original_ckpt):
    """
    Copy/paste/tweak model's weights to our SwiftFormer structure.
    """
    config = SwiftFormerConfig()
    config.num_labels = 1000
    repo_id = 'huggingface/label-files'
    filename = 'imagenet-1k-id2label.json'
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    if swiftformer_name == 'swiftformer_xs':
        config.depths = [3, 3, 6, 4]
        config.embed_dims = [48, 56, 112, 220]
    elif swiftformer_name == 'swiftformer_s':
        config.depths = [3, 3, 9, 6]
        config.embed_dims = [48, 64, 168, 224]
    elif swiftformer_name == 'swiftformer_l1':
        config.depths = [4, 3, 10, 5]
        config.embed_dims = [48, 96, 192, 384]
    elif swiftformer_name == 'swiftformer_l3':
        config.depths = [4, 4, 12, 6]
        config.embed_dims = [64, 128, 320, 512]
    if original_ckpt:
        if original_ckpt.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(original_ckpt, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(original_ckpt, map_location='cpu')
    state_dict = checkpoint
    rename_keys = create_rename_keys(state_dict)
    for rename_key_src, rename_key_dest in rename_keys:
        rename_key(state_dict, rename_key_src, rename_key_dest)
    hf_model = SwiftFormerForImageClassification(config).eval()
    hf_model.load_state_dict(state_dict)
    image = prepare_img()
    processor = ViTImageProcessor.from_pretrained('preprocessor_config')
    inputs = processor(images=image, return_tensors='pt')
    timm_logits = get_expected_output(swiftformer_name)
    hf_logits = hf_model(inputs['pixel_values']).logits
    assert hf_logits.shape == torch.Size([1, 1000])
    assert torch.allclose(hf_logits[0, 0:5], timm_logits, atol=0.001)
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model {swiftformer_name} to {pytorch_dump_folder_path}')
    hf_model.save_pretrained(pytorch_dump_folder_path)