import argparse
import torch
from transformers import ChineseCLIPConfig, ChineseCLIPModel
@torch.no_grad()
def convert_chinese_clip_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    assert config_path is not None, 'Please specify the ChineseCLIP model config of the corresponding model size.'
    config = ChineseCLIPConfig.from_pretrained(config_path)
    hf_model = ChineseCLIPModel(config).eval()
    pt_weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    pt_weights = {name[7:] if name.startswith('module.') else name: value for name, value in pt_weights.items()}
    copy_text_model_and_projection(hf_model, pt_weights)
    copy_vision_model_and_projection(hf_model, pt_weights)
    hf_model.logit_scale.data = pt_weights['logit_scale'].data
    hf_model.save_pretrained(pytorch_dump_folder_path)