import argparse
import json
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from transformers import BitImageProcessor, FocalNetConfig, FocalNetForImageClassification
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
def convert_focalnet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    model_name_to_url = {'focalnet-tiny': 'https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_tiny_srf.pth', 'focalnet-tiny-lrf': 'https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_tiny_lrf.pth', 'focalnet-small': 'https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_small_srf.pth', 'focalnet-small-lrf': 'https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_small_lrf.pth', 'focalnet-base': 'https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_base_srf.pth', 'focalnet-base-lrf': 'https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_base_lrf.pth', 'focalnet-large-lrf-fl3': 'https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_large_lrf_384.pth', 'focalnet-large-lrf-fl4': 'https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_large_lrf_384_fl4.pth', 'focalnet-xlarge-lrf-fl3': 'https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_xlarge_lrf_384.pth', 'focalnet-xlarge-lrf-fl4': 'https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_xlarge_lrf_384_fl4.pth'}
    checkpoint_url = model_name_to_url[model_name]
    print('Checkpoint URL: ', checkpoint_url)
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')['model']
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    config = get_focalnet_config(model_name)
    model = FocalNetForImageClassification(config)
    model.eval()
    model.load_state_dict(state_dict)
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    processor = BitImageProcessor(do_resize=True, size={'shortest_edge': 256}, resample=PILImageResampling.BILINEAR, do_center_crop=True, crop_size=224, do_normalize=True, image_mean=IMAGENET_DEFAULT_MEAN, image_std=IMAGENET_DEFAULT_STD)
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors='pt')
    image_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    original_pixel_values = image_transforms(image).unsqueeze(0)
    assert torch.allclose(inputs.pixel_values, original_pixel_values, atol=0.0001)
    outputs = model(**inputs)
    predicted_class_idx = outputs.logits.argmax(-1).item()
    print('Predicted class:', model.config.id2label[predicted_class_idx])
    print('First values of logits:', outputs.logits[0, :3])
    if model_name == 'focalnet-tiny':
        expected_slice = torch.tensor([0.2166, -0.4368, 0.2191])
    elif model_name == 'focalnet-tiny-lrf':
        expected_slice = torch.tensor([1.1669, 0.0125, -0.1695])
    elif model_name == 'focalnet-small':
        expected_slice = torch.tensor([0.4917, -0.043, 0.1341])
    elif model_name == 'focalnet-small-lrf':
        expected_slice = torch.tensor([-0.2588, -0.5342, -0.2331])
    elif model_name == 'focalnet-base':
        expected_slice = torch.tensor([-0.1655, -0.409, -0.173])
    elif model_name == 'focalnet-base-lrf':
        expected_slice = torch.tensor([0.5306, -0.0483, -0.3928])
    assert torch.allclose(outputs.logits[0, :3], expected_slice, atol=0.0001)
    print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        print(f'Saving model and processor of {model_name} to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print(f'Pushing model and processor of {model_name} to the hub...')
        model.push_to_hub(f'{model_name}')
        processor.push_to_hub(f'{model_name}')