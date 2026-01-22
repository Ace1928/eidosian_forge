import argparse
import os
import torch
from transformers import FlavaConfig, FlavaForPreTraining
from transformers.models.flava.convert_dalle_to_flava_codebook import convert_dalle_checkpoint
@torch.no_grad()
def convert_flava_checkpoint(checkpoint_path, codebook_path, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = FlavaConfig.from_pretrained(config_path)
    else:
        config = FlavaConfig()
    hf_model = FlavaForPreTraining(config).eval()
    codebook_state_dict = convert_dalle_checkpoint(codebook_path, None, save_checkpoint=False)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    else:
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_path, map_location='cpu')
    hf_state_dict = upgrade_state_dict(state_dict, codebook_state_dict)
    hf_model.load_state_dict(hf_state_dict)
    hf_state_dict = hf_model.state_dict()
    hf_count = count_parameters(hf_state_dict)
    state_dict_count = count_parameters(state_dict) + count_parameters(codebook_state_dict)
    assert torch.allclose(hf_count, state_dict_count, atol=0.001)
    hf_model.save_pretrained(pytorch_dump_folder_path)