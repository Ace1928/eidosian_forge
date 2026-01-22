import argparse
import json
import re
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging
@torch.no_grad()
def convert_movilevit_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our MobileNetV1 structure.
    """
    config = get_mobilenet_v1_config(model_name)
    model = MobileNetV1ForImageClassification(config).eval()
    load_tf_weights_in_mobilenet_v1(model, config, checkpoint_path)
    image_processor = MobileNetV1ImageProcessor(crop_size={'width': config.image_size, 'height': config.image_size}, size={'shortest_edge': config.image_size + 32})
    encoding = image_processor(images=prepare_img(), return_tensors='pt')
    outputs = model(**encoding)
    logits = outputs.logits
    assert logits.shape == (1, 1001)
    if model_name == 'mobilenet_v1_1.0_224':
        expected_logits = torch.tensor([-4.1739, -1.1233, 3.1205])
    elif model_name == 'mobilenet_v1_0.75_192':
        expected_logits = torch.tensor([-3.944, -2.3141, -0.3333])
    else:
        expected_logits = None
    if expected_logits is not None:
        assert torch.allclose(logits[0, :3], expected_logits, atol=0.0001)
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model {model_name} to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print('Pushing to the hub...')
        repo_id = 'google/' + model_name
        image_processor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)