import argparse
import json
from pathlib import Path
import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, Swinv2Config, Swinv2ForImageClassification
def convert_swinv2_checkpoint(swinv2_name, pytorch_dump_folder_path):
    timm_model = timm.create_model(swinv2_name, pretrained=True)
    timm_model.eval()
    config = get_swinv2_config(swinv2_name)
    model = Swinv2ForImageClassification(config)
    model.eval()
    new_state_dict = convert_state_dict(timm_model.state_dict(), model)
    model.load_state_dict(new_state_dict)
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image_processor = AutoImageProcessor.from_pretrained('microsoft/{}'.format(swinv2_name.replace('_', '-')))
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = image_processor(images=image, return_tensors='pt')
    timm_outs = timm_model(inputs['pixel_values'])
    hf_outs = model(**inputs).logits
    assert torch.allclose(timm_outs, hf_outs, atol=0.001)
    print(f'Saving model {swinv2_name} to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)
    model.push_to_hub(repo_path_or_name=Path(pytorch_dump_folder_path, swinv2_name), organization='nandwalritik', commit_message='Add model')