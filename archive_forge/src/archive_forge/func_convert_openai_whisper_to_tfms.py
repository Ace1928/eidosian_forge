import argparse
import io
import json
import os
import tempfile
import urllib
import warnings
from typing import Any, Optional, Tuple
import torch
from huggingface_hub.utils import insecure_hashlib
from torch import nn
from tqdm import tqdm
from transformers import (
from transformers.models.whisper.tokenization_whisper import LANGUAGES, bytes_to_unicode
from transformers.utils.import_utils import _is_package_available
def convert_openai_whisper_to_tfms(checkpoint_path, pytorch_dump_folder_path) -> Tuple[WhisperForConditionalGeneration, bool, int]:
    if '.pt' not in checkpoint_path:
        root = os.path.dirname(pytorch_dump_folder_path) or '.'
        original_checkpoint = _download(_MODELS[checkpoint_path], root)
        openai_version = checkpoint_path
    else:
        original_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        openai_version = None
    dimensions = original_checkpoint['dims']
    state_dict = original_checkpoint['model_state_dict']
    proj_out_weights = state_dict['decoder.token_embedding.weight']
    remove_ignore_keys_(state_dict)
    rename_keys(state_dict)
    tie_embeds = True
    ffn_dim = state_dict['decoder.layers.0.fc1.weight'].shape[0]
    endoftext_id = 50257 if dimensions['n_vocab'] > 51865 else 50256
    config = WhisperConfig(vocab_size=dimensions['n_vocab'], encoder_ffn_dim=ffn_dim, decoder_ffn_dim=ffn_dim, num_mel_bins=dimensions['n_mels'], d_model=dimensions['n_audio_state'], max_target_positions=dimensions['n_text_ctx'], encoder_layers=dimensions['n_audio_layer'], encoder_attention_heads=dimensions['n_audio_head'], decoder_layers=dimensions['n_text_layer'], decoder_attention_heads=dimensions['n_text_head'], max_source_positions=dimensions['n_audio_ctx'], eos_token_id=endoftext_id, bos_token_id=endoftext_id, pad_token_id=endoftext_id, decoder_start_token_id=endoftext_id + 1)
    model = WhisperForConditionalGeneration(config)
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 and (not set(missing) <= {'encoder.embed_positions.weights', 'decoder.embed_positions.weights'}):
        raise ValueError(f'Only `encoder.embed_positions.weights` and `decoder.embed_positions.weights`  are allowed to be missing, but all the following weights are missing {missing}')
    if tie_embeds:
        model.proj_out = make_linear_from_emb(model.model.decoder.embed_tokens)
    else:
        model.proj_out.weight.data = proj_out_weights
    is_multilingual = model.config.vocab_size >= 51865
    num_languages = model.config.vocab_size - 51765 - int(is_multilingual)
    model.generation_config = _get_generation_config(is_multilingual, num_languages, openai_version)
    return (model, is_multilingual, num_languages)