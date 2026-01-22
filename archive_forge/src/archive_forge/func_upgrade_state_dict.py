import argparse
import os
import torch
from transformers import FlavaConfig, FlavaForPreTraining
from transformers.models.flava.convert_dalle_to_flava_codebook import convert_dalle_checkpoint
def upgrade_state_dict(state_dict, codebook_state_dict):
    upgrade = {}
    for key, value in state_dict.items():
        if 'text_encoder.embeddings' in key or 'image_encoder.embeddings' in key:
            continue
        key = key.replace('heads.cmd.mim_head.cls.predictions', 'mmm_image_head')
        key = key.replace('heads.cmd.mlm_head.cls.predictions', 'mmm_text_head')
        key = key.replace('heads.cmd.itm_head.cls', 'itm_head')
        key = key.replace('heads.cmd.itm_head.pooler', 'itm_head.pooler')
        key = key.replace('heads.cmd.clip_head.logit_scale', 'flava.logit_scale')
        key = key.replace('heads.fairseq_mlm.cls.predictions', 'mlm_head')
        key = key.replace('heads.imagenet.mim_head.cls.predictions', 'mim_head')
        key = key.replace('mm_text_projection', 'flava.text_to_mm_projection')
        key = key.replace('mm_image_projection', 'flava.image_to_mm_projection')
        key = key.replace('image_encoder.module', 'flava.image_model')
        key = key.replace('text_encoder.module', 'flava.text_model')
        key = key.replace('mm_encoder.module.encoder.cls_token', 'flava.multimodal_model.cls_token')
        key = key.replace('mm_encoder.module', 'flava.multimodal_model')
        key = key.replace('text_projection', 'flava.text_projection')
        key = key.replace('image_projection', 'flava.image_projection')
        upgrade[key] = value.float()
    for key, value in codebook_state_dict.items():
        upgrade[f'image_codebook.{key}'] = value
    return upgrade