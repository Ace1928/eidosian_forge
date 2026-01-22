import argparse
import os
import torch
from transformers import FlavaConfig, FlavaForPreTraining
from transformers.models.flava.convert_dalle_to_flava_codebook import convert_dalle_checkpoint
def count_parameters(state_dict):
    return sum((param.float().sum() if 'encoder.embeddings' not in key else 0 for key, param in state_dict.items()))