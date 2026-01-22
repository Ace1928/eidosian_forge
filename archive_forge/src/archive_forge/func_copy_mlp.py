import argparse
import torch
from transformers import ChineseCLIPConfig, ChineseCLIPModel
def copy_mlp(hf_mlp, pt_weights, prefix):
    copy_linear(hf_mlp.fc1, pt_weights, f'{prefix}.c_fc')
    copy_linear(hf_mlp.fc2, pt_weights, f'{prefix}.c_proj')