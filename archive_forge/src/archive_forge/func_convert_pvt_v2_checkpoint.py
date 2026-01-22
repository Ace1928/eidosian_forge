import argparse
from pathlib import Path
import requests
import torch
from PIL import Image
from transformers import PvtImageProcessor, PvtV2Config, PvtV2ForImageClassification
from transformers.utils import logging
@torch.no_grad()
def convert_pvt_v2_checkpoint(pvt_v2_size, pvt_v2_checkpoint, pytorch_dump_folder_path, verify_imagenet_weights=False):
    """
    Copy/paste/tweak model's weights to our PVT structure.
    """
    if pvt_v2_size == 'b0':
        config_path = 'OpenGVLab/pvt_v2_b0'
    elif pvt_v2_size == 'b1':
        config_path = 'OpenGVLab/pvt_v2_b1'
    elif pvt_v2_size == 'b2':
        config_path = 'OpenGVLab/pvt_v2_b2'
    elif pvt_v2_size == 'b2-linear':
        config_path = 'OpenGVLab/pvt_v2_b2_linear'
    elif pvt_v2_size == 'b3':
        config_path = 'OpenGVLab/pvt_v2_b3'
    elif pvt_v2_size == 'b4':
        config_path = 'OpenGVLab/pvt_v2_b4'
    elif pvt_v2_size == 'b5':
        config_path = 'OpenGVLab/pvt_v2_b5'
    else:
        raise ValueError(f"Available model sizes: 'b0', 'b1', 'b2', 'b2-linear', 'b3', 'b4', 'b5', but '{pvt_v2_size}' was given")
    config = PvtV2Config.from_pretrained(config_path)
    state_dict = torch.load(pvt_v2_checkpoint, map_location='cpu')
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_k_v(state_dict, config)
    model = PvtV2ForImageClassification(config).eval()
    model.load_state_dict(state_dict)
    image_processor = PvtImageProcessor(size=config.image_size)
    if verify_imagenet_weights:
        print('Verifying conversion of pretrained ImageNet weights...')
        encoding = image_processor(images=prepare_img(), return_tensors='pt')
        pixel_values = encoding['pixel_values']
        outputs = model(pixel_values)
        logits = outputs.logits.detach().cpu()
        if pvt_v2_size == 'b0':
            expected_slice_logits = torch.tensor([-1.1939, -1.4547, -0.1076])
        elif pvt_v2_size == 'b1':
            expected_slice_logits = torch.tensor([-0.4716, -0.7335, -0.46])
        elif pvt_v2_size == 'b2':
            expected_slice_logits = torch.tensor([0.0795, -0.317, 0.2247])
        elif pvt_v2_size == 'b2-linear':
            expected_slice_logits = torch.tensor([0.0968, 0.3937, -0.4252])
        elif pvt_v2_size == 'b3':
            expected_slice_logits = torch.tensor([-0.4595, -0.287, 0.094])
        elif pvt_v2_size == 'b4':
            expected_slice_logits = torch.tensor([-0.1769, -0.1747, -0.0143])
        elif pvt_v2_size == 'b5':
            expected_slice_logits = torch.tensor([-0.2943, -0.1008, 0.6812])
        else:
            raise ValueError(f"Available model sizes: 'b0', 'b1', 'b2', 'b2-linear', 'b3', 'b4', 'b5', but '{pvt_v2_size}' was given")
        assert torch.allclose(logits[0, :3], expected_slice_logits, atol=0.0001), 'ImageNet weights not converted successfully.'
        print('ImageNet weights verified, conversion successful.')
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model pytorch_model.bin to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)