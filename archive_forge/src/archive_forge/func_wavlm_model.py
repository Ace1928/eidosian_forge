import math
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from . import components
def wavlm_model(extractor_mode: str, extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]], extractor_conv_bias: bool, encoder_embed_dim: int, encoder_projection_dropout: float, encoder_pos_conv_kernel: int, encoder_pos_conv_groups: int, encoder_num_layers: int, encoder_num_heads: int, encoder_num_buckets: int, encoder_max_distance: int, encoder_attention_dropout: float, encoder_ff_interm_features: int, encoder_ff_interm_dropout: float, encoder_dropout: float, encoder_layer_norm_first: bool, encoder_layer_drop: float, aux_num_out: Optional[int]) -> Wav2Vec2Model:
    """Builds custom WaveLM model :cite:`chen2022wavlm`. The architecture is compatible
    with Wav2Vec2 model :cite:`baevski2020wav2vec`, and so the output object is
    :class:`~torchaudio.models.Wav2Vec2Model`. Most of the arguments have the same meaning
    as in :py:func:`~torchaudio.models.wav2vec2_model` so please refer there for documentation.

    Args:
        extractor_mode (str): Operation mode of feature extractor.
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        extractor_conv_layer_config (list of integer tuples or None):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        extractor_conv_bias (bool):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        encoder_embed_dim (int):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        encoder_projection_dropout (float):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        encoder_pos_conv_kernel (int):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        encoder_pos_conv_groups (int):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        encoder_num_layers (int):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        encoder_num_heads (int):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        encoder_num_buckets (int):
            Number of buckets for relative position embedding.
        encoder_max_distance (int):
            Maximum distance for relative position embedding.

        encoder_attention_dropout (float):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        encoder_ff_interm_features (int):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        encoder_ff_interm_dropout (float):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        encoder_dropout (float):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        encoder_layer_norm_first (bool):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        encoder_layer_drop (float):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

        aux_num_out (int or None):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
    feature_extractor = components._get_feature_extractor(extractor_mode, extractor_conv_layer_config, extractor_conv_bias)
    encoder = components._get_wavlm_encoder(in_features=extractor_conv_layer_config[-1][0], embed_dim=encoder_embed_dim, dropout_input=encoder_projection_dropout, pos_conv_kernel=encoder_pos_conv_kernel, pos_conv_groups=encoder_pos_conv_groups, num_layers=encoder_num_layers, num_heads=encoder_num_heads, num_buckets=encoder_num_buckets, max_distance=encoder_max_distance, attention_dropout=encoder_attention_dropout, ff_interm_features=encoder_ff_interm_features, ff_interm_dropout=encoder_ff_interm_dropout, dropout=encoder_dropout, layer_norm_first=encoder_layer_norm_first, layer_drop=encoder_layer_drop)
    aux = None
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_embed_dim, out_features=aux_num_out)
    return Wav2Vec2Model(feature_extractor, encoder, aux)