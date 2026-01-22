import argparse
import json
from collections import OrderedDict
import torch
from huggingface_hub import cached_download, hf_hub_url
from transformers import AutoImageProcessor, CvtConfig, CvtForImageClassification
def embeddings(idx):
    """
    The function helps in renaming embedding layer weights.

    Args:
        idx: stage number in original model
    """
    embed = []
    embed.append((f'cvt.encoder.stages.{idx}.embedding.convolution_embeddings.projection.weight', f'stage{idx}.patch_embed.proj.weight'))
    embed.append((f'cvt.encoder.stages.{idx}.embedding.convolution_embeddings.projection.bias', f'stage{idx}.patch_embed.proj.bias'))
    embed.append((f'cvt.encoder.stages.{idx}.embedding.convolution_embeddings.normalization.weight', f'stage{idx}.patch_embed.norm.weight'))
    embed.append((f'cvt.encoder.stages.{idx}.embedding.convolution_embeddings.normalization.bias', f'stage{idx}.patch_embed.norm.bias'))
    return embed