import argparse
import torch
from transformers import UnivNetConfig, UnivNetModel, logging
def convert_univnet_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, repo_id=None, safe_serialization=False):
    model_state_dict_base = torch.load(checkpoint_path, map_location='cpu')
    state_dict = model_state_dict_base['model_g']
    if config_path is not None:
        config = UnivNetConfig.from_pretrained(config_path)
    else:
        config = UnivNetConfig()
    keys_to_modify = get_key_mapping(config)
    keys_to_remove = set()
    hf_state_dict = rename_state_dict(state_dict, keys_to_modify, keys_to_remove)
    model = UnivNetModel(config)
    model.apply_weight_norm()
    model.load_state_dict(hf_state_dict)
    model.remove_weight_norm()
    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=safe_serialization)
    if repo_id:
        print('Pushing to the hub...')
        model.push_to_hub(repo_id)