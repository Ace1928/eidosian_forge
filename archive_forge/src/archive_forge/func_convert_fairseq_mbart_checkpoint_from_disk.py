import argparse
import torch
from torch import nn
from transformers import MBartConfig, MBartForConditionalGeneration
def convert_fairseq_mbart_checkpoint_from_disk(checkpoint_path, hf_config_path='facebook/mbart-large-en-ro', finetuned=False, mbart_50=False):
    state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
    remove_ignore_keys_(state_dict)
    vocab_size = state_dict['encoder.embed_tokens.weight'].shape[0]
    mbart_config = MBartConfig.from_pretrained(hf_config_path, vocab_size=vocab_size)
    if mbart_50 and finetuned:
        mbart_config.activation_function = 'relu'
    state_dict['shared.weight'] = state_dict['decoder.embed_tokens.weight']
    model = MBartForConditionalGeneration(mbart_config)
    model.model.load_state_dict(state_dict)
    if finetuned:
        model.lm_head = make_linear_from_emb(model.model.shared)
    return model