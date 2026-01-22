import argparse
from pathlib import Path
import torch
import yaml
from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig, logging
def remap_hifigan_yaml_config(yaml_config_path):
    with Path(yaml_config_path).open('r', encoding='utf-8') as f:
        args = yaml.safe_load(f)
        args = argparse.Namespace(**args)
    vocoder_type = args.tts_conf['vocoder_type']
    if vocoder_type != 'hifigan_generator':
        raise TypeError(f'Vocoder config must be for `hifigan_generator`, but got {vocoder_type}')
    remapped_dict = {}
    vocoder_params = args.tts_conf['vocoder_params']
    key_mappings = {'channels': 'upsample_initial_channel', 'in_channels': 'model_in_dim', 'resblock_dilations': 'resblock_dilation_sizes', 'resblock_kernel_sizes': 'resblock_kernel_sizes', 'upsample_kernel_sizes': 'upsample_kernel_sizes', 'upsample_scales': 'upsample_rates'}
    for espnet_config_key, hf_config_key in key_mappings.items():
        remapped_dict[hf_config_key] = vocoder_params[espnet_config_key]
    remapped_dict['sampling_rate'] = args.tts_conf['sampling_rate']
    remapped_dict['normalize_before'] = False
    remapped_dict['leaky_relu_slope'] = vocoder_params['nonlinear_activation_params']['negative_slope']
    return remapped_dict