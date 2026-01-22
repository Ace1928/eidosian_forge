import argparse
import torch
from transformers import NystromformerConfig, NystromformerForMaskedLM
def convert_nystromformer_checkpoint(checkpoint_path, nystromformer_config_file, pytorch_dump_path):
    orig_state_dict = torch.load(checkpoint_path, map_location='cpu')['model_state_dict']
    config = NystromformerConfig.from_json_file(nystromformer_config_file)
    model = NystromformerForMaskedLM(config)
    new_state_dict = convert_checkpoint_helper(config, orig_state_dict)
    model.load_state_dict(new_state_dict)
    model.eval()
    model.save_pretrained(pytorch_dump_path)
    print(f'Checkpoint successfuly converted. Model saved at {pytorch_dump_path}')