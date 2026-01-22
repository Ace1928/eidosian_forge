import argparse
import os
import torch
from huggingface_hub import hf_hub_download
from transformers import ClvpConfig, ClvpModelForConditionalGeneration
def convert_clvp_weights(checkpoint_path, pytorch_dump_folder_path):
    converted_checkpoint = {}
    for each_model_name, each_model_url in _MODELS.items():
        each_model_path = os.path.join(checkpoint_path, each_model_url.split('/')[-1])
        if not os.path.exists(each_model_path):
            print(f'\n{each_model_name} was not found! Downloading it to {each_model_path}')
            _download(url=each_model_url, root=each_model_path)
        if each_model_name == 'clvp':
            clvp_checkpoint = torch.load(each_model_path, map_location='cpu')
        else:
            decoder_checkpoint = torch.load(each_model_path, map_location='cpu')
    converted_checkpoint.update(**convert_encoder_weights(clvp_checkpoint))
    converted_checkpoint.update(**convert_decoder_weights(decoder_checkpoint))
    config = ClvpConfig.from_pretrained('susnato/clvp_dev')
    model = ClvpModelForConditionalGeneration(config)
    model.load_state_dict(converted_checkpoint, strict=True)
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Model saved at {pytorch_dump_folder_path}!')