import argparse
import torch
from transformers import ChineseCLIPConfig, ChineseCLIPModel
def copy_attn_layer(hf_attn_layer, pt_weights, prefix):
    q_proj, k_proj, v_proj = pt_weights[f'{prefix}.in_proj_weight'].chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = pt_weights[f'{prefix}.in_proj_bias'].chunk(3, dim=0)
    out_proj_weights = pt_weights[f'{prefix}.out_proj.weight']
    out_proj_bias = pt_weights[f'{prefix}.out_proj.bias']
    hf_attn_layer.q_proj.weight.data = q_proj
    hf_attn_layer.q_proj.bias.data = q_proj_bias
    hf_attn_layer.k_proj.weight.data = k_proj
    hf_attn_layer.k_proj.bias.data = k_proj_bias
    hf_attn_layer.v_proj.weight.data = v_proj
    hf_attn_layer.v_proj.bias.data = v_proj_bias
    hf_attn_layer.out_proj.weight.data = out_proj_weights
    hf_attn_layer.out_proj.bias.data = out_proj_bias