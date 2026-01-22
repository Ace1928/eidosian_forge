import argparse
import json
from pathlib import Path
import requests
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from transformers import BitImageProcessor, Dinov2Config, Dinov2ForImageClassification, Dinov2Model
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from transformers.utils import logging
@torch.no_grad()
def convert_dinov2_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our DINOv2 structure.
    """
    image_classifier = '1layer' in model_name
    config = get_dinov2_config(model_name, image_classifier=image_classifier)
    original_model = torch.hub.load('facebookresearch/dinov2', model_name.replace('_1layer', ''))
    original_model.eval()
    state_dict = original_model.state_dict()
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config)
    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if 'w12' in key:
            key = key.replace('w12', 'weights_in')
        if 'w3' in key:
            key = key.replace('w3', 'weights_out')
        state_dict[key] = val
    if image_classifier:
        model = Dinov2ForImageClassification(config).eval()
        model.dinov2.load_state_dict(state_dict)
        model_name_to_classifier_dict_url = {'dinov2_vits14_1layer': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_linear_head.pth', 'dinov2_vitb14_1layer': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_linear_head.pth', 'dinov2_vitl14_1layer': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth', 'dinov2_vitg14_1layer': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth'}
        url = model_name_to_classifier_dict_url[model_name]
        classifier_state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.classifier.weight = nn.Parameter(classifier_state_dict['weight'])
        model.classifier.bias = nn.Parameter(classifier_state_dict['bias'])
    else:
        model = Dinov2Model(config).eval()
        model.load_state_dict(state_dict)
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    transformations = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
    original_pixel_values = transformations(image).unsqueeze(0)
    processor = BitImageProcessor(size={'shortest_edge': 256}, resample=PILImageResampling.BICUBIC, image_mean=IMAGENET_DEFAULT_MEAN, image_std=IMAGENET_DEFAULT_STD)
    pixel_values = processor(image, return_tensors='pt').pixel_values
    assert torch.allclose(original_pixel_values, pixel_values)
    with torch.no_grad():
        outputs = model(pixel_values, output_hidden_states=True)
        original_outputs = original_model(pixel_values)
    if image_classifier:
        print('Predicted class:')
        class_idx = outputs.logits.argmax(-1).item()
        print(model.config.id2label[class_idx])
    else:
        assert outputs.last_hidden_state[:, 0].shape == original_outputs.shape
        assert torch.allclose(outputs.last_hidden_state[:, 0], original_outputs, atol=0.001)
    print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f'Saving model {model_name} to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        print(f'Saving image processor to {pytorch_dump_folder_path}')
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        model_name_to_hf_name = {'dinov2_vits14': 'dinov2-small', 'dinov2_vitb14': 'dinov2-base', 'dinov2_vitl14': 'dinov2-large', 'dinov2_vitg14': 'dinov2-giant', 'dinov2_vits14_1layer': 'dinov2-small-imagenet1k-1-layer', 'dinov2_vitb14_1layer': 'dinov2-base-imagenet1k-1-layer', 'dinov2_vitl14_1layer': 'dinov2-large-imagenet1k-1-layer', 'dinov2_vitg14_1layer': 'dinov2-giant-imagenet1k-1-layer'}
        name = model_name_to_hf_name[model_name]
        model.push_to_hub(f'facebook/{name}')
        processor.push_to_hub(f'facebook/{name}')