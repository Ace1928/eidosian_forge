import argparse
import torch
from torch import nn
from transformers import PLBartConfig, PLBartForConditionalGeneration, PLBartForSequenceClassification
def convert_fairseq_plbart_checkpoint_from_disk(checkpoint_path, hf_config_path='uclanlp/plbart-base', finetuned=False, classification=False):
    state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
    remove_ignore_keys_(state_dict)
    vocab_size = state_dict['encoder.embed_tokens.weight'].shape[0]
    plbart_config = PLBartConfig.from_pretrained(hf_config_path, vocab_size=vocab_size)
    state_dict['shared.weight'] = state_dict['decoder.embed_tokens.weight']
    if not classification:
        model = PLBartForConditionalGeneration(plbart_config)
        model.model.load_state_dict(state_dict)
        if finetuned:
            model.lm_head = make_linear_from_emb(model.model.shared)
    else:
        classification_head = {}
        for key, value in state_dict.copy().items():
            if key.startswith('classification_heads.sentence_classification_head'):
                classification_head[key.replace('classification_heads.sentence_classification_head.', '')] = value
                state_dict.pop(key)
        model = PLBartForSequenceClassification(plbart_config)
        model.model.load_state_dict(state_dict)
        model.classification_head.load_state_dict(classification_head)
    return model