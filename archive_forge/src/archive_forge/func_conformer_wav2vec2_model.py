from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.conformer import ConformerLayer
from torchaudio.models.rnnt import _TimeReduction
from torchaudio.models.wav2vec2 import components
def conformer_wav2vec2_model(extractor_input_dim: int, extractor_output_dim: int, extractor_stride: int, encoder_embed_dim: int, encoder_projection_dropout: float, encoder_num_layers: int, encoder_num_heads: int, encoder_ff_interm_features: int, encoder_depthwise_conv_kernel_size: Union[int, List[int]], encoder_dropout: float, encoder_convolution_first: bool, encoder_use_group_norm: bool) -> Wav2Vec2Model:
    """Build a custom Conformer Wav2Vec2Model

    Args:
        extractor_input_dim (int): Input dimension of the features.
        extractor_output_dim (int): Output dimension after feature extraction.
        extractor_stride (int): Stride used in time reduction layer of feature extraction.
        encoder_embed_dim (int): The dimension of the embedding in the feature projection.
        encoder_projection_dropout (float):
            The dropout probability applied after the input feature is projected to ``embed_dim``
        encoder_num_layers (int): Number of Conformer layers in the encoder.
        encoder_num_heads (int): Number of heads in each Conformer layer.
        encoder_ff_interm_features (int):
            Hidden layer dimension of the feedforward network in each Conformer layer.
        encoder_depthwise_conv_kernel_size (int or List[int]):
            List of kernel sizes corresponding to each of the Conformer layers.
            If int is provided, all layers will have the same kernel size.
        encoder_dropout (float): Dropout probability in each Conformer layer.
        encoder_convolution_first (bool):
            Whether to apply the convolution module ahead of the attention module
            in each Conformer layer.
        encoder_use_group_norm (bool):
            Whether to use ``GroupNorm`` rather than ``BatchNorm1d`` in the convolution
            module in each Conformer layer.

    Returns:
        Wav2Vec2Model:
            The resulting wav2vec2 model with a conformer encoder.
    """
    feature_extractor = _get_conformer_feature_extractor(extractor_input_dim, extractor_output_dim, extractor_stride)
    encoder = _get_conformer_encoder(in_features=extractor_output_dim, embed_dim=encoder_embed_dim, dropout_input=encoder_projection_dropout, num_layers=encoder_num_layers, num_heads=encoder_num_heads, ff_interm_features=encoder_ff_interm_features, depthwise_conv_kernel_size=encoder_depthwise_conv_kernel_size, dropout=encoder_dropout, convolution_first=encoder_convolution_first, use_group_norm=encoder_use_group_norm)
    return Wav2Vec2Model(feature_extractor, encoder)