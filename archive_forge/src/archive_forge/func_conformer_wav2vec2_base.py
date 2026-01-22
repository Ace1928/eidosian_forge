from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.conformer import ConformerLayer
from torchaudio.models.rnnt import _TimeReduction
from torchaudio.models.wav2vec2 import components
def conformer_wav2vec2_base(extractor_input_dim: int=64, extractor_output_dim: int=256, encoder_projection_dropout: float=0.0) -> Wav2Vec2Model:
    """
    Build Conformer Wav2Vec2 Model with "small" architecture from
    *Conformer-Based Slef-Supervised Learning for Non-Speech Audio Tasks* :cite:`9746490`

    Args:
        extractor_input_dim (int, optional): Input dimension of feature extractor. (Default: 64)
        extractor_output_dim (int, optional): Output dimension of feature extractor. (Default: 256)
        encoder_projection_dropout (float, optional):
            Dropout probability applied after feature projection. (Default: 0.0)

    Returns:
        Wav2Vec2Model:
            The resulting wav2vec2 model with a conformer encoder and ``base`` configuration.
    """
    return conformer_wav2vec2_model(extractor_input_dim=extractor_input_dim, extractor_output_dim=extractor_output_dim, extractor_stride=4, encoder_embed_dim=256, encoder_projection_dropout=encoder_projection_dropout, encoder_num_layers=12, encoder_num_heads=8, encoder_ff_interm_features=1024, encoder_depthwise_conv_kernel_size=[31] + [15] * 11, encoder_dropout=0.1, encoder_convolution_first=True, encoder_use_group_norm=True)