import argparse
import torch
from torch import nn
from transformers import M2M100Config, M2M100ForConditionalGeneration
def convert_fairseq_m2m100_checkpoint_from_disk(checkpoint_path):
    m2m_100 = torch.load(checkpoint_path, map_location='cpu')
    args = m2m_100['args'] or m2m_100['cfg']['model']
    state_dict = m2m_100['model']
    remove_ignore_keys_(state_dict)
    vocab_size = state_dict['encoder.embed_tokens.weight'].shape[0]
    config = M2M100Config(vocab_size=vocab_size, max_position_embeddings=1024, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, encoder_attention_heads=args.encoder_attention_heads, decoder_attention_heads=args.decoder_attention_heads, encoder_ffn_dim=args.encoder_ffn_embed_dim, decoder_ffn_dim=args.decoder_ffn_embed_dim, d_model=args.encoder_embed_dim, encoder_layerdrop=args.encoder_layerdrop, decoder_layerdrop=args.decoder_layerdrop, dropout=args.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, activation_function='relu')
    state_dict['shared.weight'] = state_dict['decoder.embed_tokens.weight']
    model = M2M100ForConditionalGeneration(config)
    model.model.load_state_dict(state_dict, strict=False)
    model.lm_head = make_linear_from_emb(model.model.shared)
    return model