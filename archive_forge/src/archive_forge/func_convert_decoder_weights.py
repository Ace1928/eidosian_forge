import argparse
import os
import torch
from huggingface_hub import hf_hub_download
from transformers import ClvpConfig, ClvpModelForConditionalGeneration
def convert_decoder_weights(original_weights):
    converted_weights = {}
    original_weights_keys = sorted(original_weights.keys())
    for original_key in original_weights_keys:
        updated_key = original_key
        if len(updated_key.split('.')) > 3:
            index, attr = (updated_key.split('.')[2], updated_key.split('.')[-1])
        if 'attn.c_attn' in updated_key:
            if attr == 'weight':
                slice1, slice2, slice3 = original_weights[updated_key].squeeze(-1).T.split(split_size=dim, dim=0)
            else:
                slice1, slice2, slice3 = original_weights[updated_key].split(split_size=dim, dim=0)
            converted_weights[f'speech_decoder_model.model.decoder.layers.{index}.attn.q_proj.{attr}'] = slice1
            converted_weights[f'speech_decoder_model.model.decoder.layers.{index}.attn.k_proj.{attr}'] = slice2
            converted_weights[f'speech_decoder_model.model.decoder.layers.{index}.attn.v_proj.{attr}'] = slice3
            continue
        if 'attn.c_proj' in updated_key:
            converted_weights[f'speech_decoder_model.model.decoder.layers.{index}.attn.out_proj.{attr}'] = original_weights[updated_key].squeeze(-1).T
            continue
        if 'attn.bias' in updated_key or 'attn.masked_bias' in updated_key or 'text_head' in updated_key:
            original_weights.pop(updated_key)
            continue
        if 'qkv' in updated_key:
            if attr == 'weight':
                slice1, slice2, slice3 = original_weights[updated_key].squeeze(-1).split(split_size=dim, dim=0)
            else:
                slice1, slice2, slice3 = original_weights[updated_key].split(split_size=dim, dim=0)
            indices = torch.arange(dim)
            index1, index2, index3 = (indices.unfold(0, sub_dim, sub_dim * 3).flatten(), indices[sub_dim:].unfold(0, sub_dim, sub_dim * 3).flatten(), indices[2 * sub_dim:].unfold(0, sub_dim, sub_dim * 3).flatten())
            converted_weights[f'conditioning_encoder.mel_attn_blocks.{index}.q_proj.{attr}'] = torch.concatenate([slice1[index1], slice2[index3], slice3[index2]], axis=0)
            converted_weights[f'conditioning_encoder.mel_attn_blocks.{index}.k_proj.{attr}'] = torch.concatenate([slice1[index2], slice2[index1], slice3[index3]], axis=0)
            converted_weights[f'conditioning_encoder.mel_attn_blocks.{index}.v_proj.{attr}'] = torch.concatenate([slice1[index3], slice2[index2], slice3[index1]], axis=0)
            continue
        if 'proj_out' in updated_key:
            converted_weights[f'conditioning_encoder.mel_attn_blocks.{index}.out_proj.{attr}'] = original_weights[updated_key].squeeze(-1)
            continue
        for k, v in CLVP_DECODER_MAPPING.items():
            if k in updated_key:
                updated_key = updated_key.replace(k, v)
        converted_weights[updated_key] = original_weights.pop(original_key)
    return converted_weights