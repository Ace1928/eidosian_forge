import argparse
import json
import os
import fairseq
import torch
from fairseq.data import Dictionary
from sew_asapp import tasks  # noqa: F401
from transformers import (
def convert_config(model, is_finetuned):
    config = SEWDConfig()
    if is_finetuned:
        fs_config = model.w2v_encoder.w2v_model.cfg
    else:
        fs_config = model.cfg
    config.conv_bias = fs_config.conv_bias
    conv_layers = eval(fs_config.conv_feature_layers)
    config.conv_dim = [x[0] for x in conv_layers]
    config.conv_kernel = [x[1] for x in conv_layers]
    config.conv_stride = [x[2] for x in conv_layers]
    config.feat_extract_activation = 'gelu'
    config.feat_extract_norm = 'layer' if fs_config.extractor_mode == 'layer_norm' else 'group'
    config.final_dropout = 0.0
    config.hidden_act = fs_config.activation_fn.name
    config.hidden_size = fs_config.encoder_embed_dim
    config.initializer_range = 0.02
    config.intermediate_size = fs_config.encoder_ffn_embed_dim
    config.layer_norm_eps = 1e-05
    config.layerdrop = fs_config.encoder_layerdrop
    config.num_attention_heads = fs_config.encoder_attention_heads
    config.num_conv_pos_embedding_groups = fs_config.conv_pos_groups
    config.num_conv_pos_embeddings = fs_config.conv_pos
    config.num_feat_extract_layers = len(conv_layers)
    config.num_hidden_layers = fs_config.encoder_layers
    config.squeeze_factor = fs_config.squeeze_factor
    config.max_position_embeddings = fs_config.max_position_embeddings
    config.position_buckets = fs_config.position_buckets
    config.share_att_key = fs_config.share_att_key
    config.relative_attention = fs_config.relative_attention
    config.position_biased_input = fs_config.position_biased_input
    config.pos_att_type = tuple(fs_config.pos_att_type.split('|'))
    config.norm_rel_ebd = fs_config.norm_rel_ebd
    if is_finetuned:
        fs_config = model.cfg
        config.final_dropout = fs_config.final_dropout
        config.layerdrop = fs_config.layerdrop
    config.activation_dropout = fs_config.activation_dropout
    config.apply_spec_augment = fs_config.mask_prob > 0 or fs_config.mask_channel_prob > 0
    config.attention_dropout = fs_config.attention_dropout
    config.feat_proj_dropout = fs_config.dropout_input
    config.hidden_dropout = fs_config.dropout
    config.mask_feature_length = fs_config.mask_channel_length
    config.mask_feature_prob = fs_config.mask_channel_prob
    config.mask_time_length = fs_config.mask_length
    config.mask_time_prob = fs_config.mask_prob
    config.feature_extractor_type = 'Wav2Vec2FeatureExtractor'
    config.tokenizer_class = 'Wav2Vec2CTCTokenizer'
    return config