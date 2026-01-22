import argparse
import torch
from transformers import UnivNetConfig, UnivNetModel, logging
def get_kernel_predictor_key_mapping(config: UnivNetConfig, old_prefix: str='', new_prefix: str=''):
    mapping = {}
    mapping[f'{old_prefix}.input_conv.0.weight_g'] = f'{new_prefix}.input_conv.weight_g'
    mapping[f'{old_prefix}.input_conv.0.weight_v'] = f'{new_prefix}.input_conv.weight_v'
    mapping[f'{old_prefix}.input_conv.0.bias'] = f'{new_prefix}.input_conv.bias'
    for i in range(config.kernel_predictor_num_blocks):
        mapping[f'{old_prefix}.residual_convs.{i}.1.weight_g'] = f'{new_prefix}.resblocks.{i}.conv1.weight_g'
        mapping[f'{old_prefix}.residual_convs.{i}.1.weight_v'] = f'{new_prefix}.resblocks.{i}.conv1.weight_v'
        mapping[f'{old_prefix}.residual_convs.{i}.1.bias'] = f'{new_prefix}.resblocks.{i}.conv1.bias'
        mapping[f'{old_prefix}.residual_convs.{i}.3.weight_g'] = f'{new_prefix}.resblocks.{i}.conv2.weight_g'
        mapping[f'{old_prefix}.residual_convs.{i}.3.weight_v'] = f'{new_prefix}.resblocks.{i}.conv2.weight_v'
        mapping[f'{old_prefix}.residual_convs.{i}.3.bias'] = f'{new_prefix}.resblocks.{i}.conv2.bias'
    mapping[f'{old_prefix}.kernel_conv.weight_g'] = f'{new_prefix}.kernel_conv.weight_g'
    mapping[f'{old_prefix}.kernel_conv.weight_v'] = f'{new_prefix}.kernel_conv.weight_v'
    mapping[f'{old_prefix}.kernel_conv.bias'] = f'{new_prefix}.kernel_conv.bias'
    mapping[f'{old_prefix}.bias_conv.weight_g'] = f'{new_prefix}.bias_conv.weight_g'
    mapping[f'{old_prefix}.bias_conv.weight_v'] = f'{new_prefix}.bias_conv.weight_v'
    mapping[f'{old_prefix}.bias_conv.bias'] = f'{new_prefix}.bias_conv.bias'
    return mapping