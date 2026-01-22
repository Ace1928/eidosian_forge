import argparse
import collections
from pathlib import Path
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from numpy import load
from PIL import Image
from transformers import SiglipConfig, SiglipImageProcessor, SiglipModel, SiglipProcessor, SiglipTokenizer
from transformers.utils import logging
@torch.no_grad()
def convert_siglip_checkpoint(model_name, pytorch_dump_folder_path, verify_logits=True, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our SigLIP structure.
    """
    config = get_siglip_config(model_name)
    checkpoint = model_name_to_checkpoint[model_name]
    if 'i18n' in model_name:
        vocab_file = '/Users/nielsrogge/Documents/SigLIP/multilingual_vocab/sentencepiece.model'
    else:
        vocab_file = '/Users/nielsrogge/Documents/SigLIP/english_vocab/sentencepiece.model'
    data = load(checkpoint)
    state_dict = flatten_nested_dict(data)
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest, config)
    read_in_q_k_v_head(state_dict, config)
    model = SiglipModel(config).eval()
    model.load_state_dict(state_dict)
    image_size = config.vision_config.image_size
    size = {'height': image_size, 'width': image_size}
    image_processor = SiglipImageProcessor(size=size)
    tokenizer = SiglipTokenizer(vocab_file=vocab_file, model_input_names=['input_ids'])
    processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)
    url_1 = 'https://cdn.openai.com/multimodal-neurons/assets/apple/apple-ipod.jpg'
    image_1 = Image.open(requests.get(url_1, stream=True).raw).convert('RGB')
    url_2 = 'https://cdn.openai.com/multimodal-neurons/assets/apple/apple-blank.jpg'
    image_2 = Image.open(requests.get(url_2, stream=True).raw).convert('RGB')
    texts = ['an apple', 'a picture of an apple']
    inputs = processor(images=[image_1, image_2], text=texts, return_tensors='pt', padding='max_length')
    if image_size == 224:
        filename = 'siglip_pixel_values.pt'
    elif image_size == 256:
        filename = 'siglip_pixel_values_256.pt'
    elif image_size == 384:
        filename = 'siglip_pixel_values_384.pt'
    elif image_size == 512:
        filename = 'siglip_pixel_values_512.pt'
    else:
        raise ValueError('Image size not supported')
    filepath = hf_hub_download(repo_id='nielsr/test-image', filename=filename, repo_type='dataset')
    original_pixel_values = torch.load(filepath)
    filepath = hf_hub_download(repo_id='nielsr/test-image', filename='siglip_input_ids.pt', repo_type='dataset')
    original_input_ids = torch.load(filepath)
    if 'i18n' not in model_name:
        assert inputs.input_ids.tolist() == original_input_ids.tolist()
    print('Mean of original pixel values:', original_pixel_values.mean())
    print('Mean of new pixel values:', inputs.pixel_values.mean())
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, pixel_values=original_pixel_values)
    print(outputs.logits_per_image[:3, :3])
    probs = torch.sigmoid(outputs.logits_per_image)
    print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
    print(f"{probs[0][1]:.1%} that image 0 is '{texts[1]}'")
    if verify_logits:
        if model_name == 'siglip-base-patch16-224':
            expected_slice = torch.tensor([[-2.9621, -2.1672], [-0.2713, 0.291]])
        elif model_name == 'siglip-base-patch16-256':
            expected_slice = torch.tensor([[-3.1146, -1.9894], [-0.7312, 0.6387]])
        elif model_name == 'siglip-base-patch16-384':
            expected_slice = torch.tensor([[-2.8098, -2.1891], [-0.4242, 0.4102]])
        elif model_name == 'siglip-base-patch16-512':
            expected_slice = torch.tensor([[-2.7899, -2.2668], [-0.4295, -0.0735]])
        elif model_name == 'siglip-large-patch16-256':
            expected_slice = torch.tensor([[-1.5827, -0.5801], [-0.9153, 0.1363]])
        elif model_name == 'siglip-large-patch16-384':
            expected_slice = torch.tensor([[-2.1523, -0.2899], [-0.2959, 0.7884]])
        elif model_name == 'siglip-so400m-patch14-384':
            expected_slice = torch.tensor([[-1.2441, -0.6649], [-0.706, 0.7374]])
        elif model_name == 'siglip-base-patch16-256-i18n':
            expected_slice = torch.tensor([[-0.9064, 0.1073], [-0.0299, 0.5304]])
        assert torch.allclose(outputs.logits_per_image[:3, :3], expected_slice, atol=0.0001)
        print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f'Saving model {model_name} to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        print(f'Saving processor to {pytorch_dump_folder_path}')
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        model.push_to_hub(f'nielsr/{model_name}')
        processor.push_to_hub(f'nielsr/{model_name}')