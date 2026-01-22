import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
import yaml
from transformers import (
@torch.no_grad()
def convert_FastSpeech2ConformerModel_checkpoint(checkpoint_path, yaml_config_path, pytorch_dump_folder_path, repo_id=None):
    model_params, tokenizer_name, vocab = remap_model_yaml_config(yaml_config_path)
    config = FastSpeech2ConformerConfig(**model_params)
    model = FastSpeech2ConformerModel(config)
    espnet_checkpoint = torch.load(checkpoint_path)
    hf_compatible_state_dict = convert_espnet_state_dict_to_hf(espnet_checkpoint)
    model.load_state_dict(hf_compatible_state_dict)
    model.save_pretrained(pytorch_dump_folder_path)
    with TemporaryDirectory() as tempdir:
        vocab = {token: id for id, token in enumerate(vocab)}
        vocab_file = Path(tempdir) / 'vocab.json'
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f)
        should_strip_spaces = 'no_space' in tokenizer_name
        tokenizer = FastSpeech2ConformerTokenizer(str(vocab_file), should_strip_spaces=should_strip_spaces)
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    if repo_id:
        print('Pushing to the hub...')
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)