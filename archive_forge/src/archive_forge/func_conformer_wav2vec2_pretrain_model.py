from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.conformer import ConformerLayer
from torchaudio.models.rnnt import _TimeReduction
from torchaudio.models.wav2vec2 import components
def conformer_wav2vec2_pretrain_model(extractor_input_dim: int, extractor_output_dim: int, extractor_stride: int, encoder_embed_dim: int, encoder_projection_dropout: float, encoder_num_layers: int, encoder_num_heads: int, encoder_ff_interm_features: int, encoder_depthwise_conv_kernel_size: int, encoder_dropout: float, encoder_convolution_first: bool, encoder_use_group_norm: bool, mask_prob: float, mask_selection: str, mask_other: float, mask_length: int, no_mask_overlap: bool, mask_min_space: int, mask_channel_prob: float, mask_channel_selection: str, mask_channel_other: float, mask_channel_length: int, no_mask_channel_overlap: bool, mask_channel_min_space: int, num_negatives: int, cross_sample_negatives: int) -> ConformerWav2Vec2PretrainModel:
    """Build a custom Conformer Wav2Vec2 Model for pre-training

    Args:
        extractor_input_dim (int): Input dimension of the features.
        extractor_output_dim (int): Output dimension after feature extraction.
        extractor_stride (int):
            Stride used in time reduction layer of feature extraction.
        encoder_embed_dim (int):
            The dimension of the embedding in the feature projection.
        encoder_projection_dropout (float):
            The dropout probability applied after the input feature is projected to
            ``embed_dim``
        encoder_num_layers (int):
            Number of Conformer layers in the encoder.
        encoder_num_heads (int):
            Number of heads in each Conformer layer.
        encoder_ff_interm_features (int):
            Hidden layer dimension of the feedforward network in each Conformer layer.
        encoder_depthwise_conv_kernel_size (int or List[int]):
            List of kernel sizes corresponding to each of the Conformer layers.
            If int is provided, all layers will have the same kernel size.
        encoder_dropout (float):
            Dropout probability in each Conformer layer.
        encoder_convolution_first (bool):
            Whether to apply the convolution module ahead of the attention module
            in each Conformer layer.
        encoder_use_group_norm (bool):
            Whether to use ``GroupNorm`` rather than ``BatchNorm1d`` in the convolution
            module in each Conformer layer.
        mask_prob (float):
            Probability for each token to be chosen as start of the span to be masked.
        mask_selection (str)
            How to choose the mask length. Options: [``static``, ``uniform``, ``normal``, ``poisson``].
        mask_other (float):
            Secondary mask argument (used for more complex distributions).
        mask_length (int):
            The lengths of the mask.
        no_mask_overlap (bool):
            Whether to allow masks to overlap.
        mask_min_space (int):
            Minimum space between spans (if no overlap is enabled).
        mask_channel_prob: (float):
            The probability of replacing a feature with 0.
        mask_channel_selection (str):
            How to choose the mask length for channel masking.
            Options: [``static``, ``uniform``, ``normal``, ``poisson``].
        mask_channel_other (float):
            Secondary mask argument for channel masking (used for more complex distributions).
        mask_channel_length (int):
            Minimum space between spans (if no overlap is enabled) for channel masking.
        no_mask_channel_overlap (bool):
            Whether to allow channel masks to overlap.
        mask_channel_min_space (int):
            Minimum space between spans for channel masking (if no overlap is enabled).
        num_negatives (int):
            Number of negatives to sample.
        cross_sample_negatives (int):
            Number of cross sampled negatives.

    Returns:
        ConformerWav2Vec2PretrainModel:
            The resulting model.
    """
    wav2vec2 = conformer_wav2vec2_model(extractor_input_dim, extractor_output_dim, extractor_stride, encoder_embed_dim, encoder_projection_dropout, encoder_num_layers, encoder_num_heads, encoder_ff_interm_features, encoder_depthwise_conv_kernel_size, encoder_dropout, encoder_convolution_first, encoder_use_group_norm)
    mask_generator = components.MaskGenerator(extractor_output_dim, mask_prob, mask_selection, mask_other, mask_length, no_mask_overlap, mask_min_space, mask_channel_prob, mask_channel_selection, mask_channel_other, mask_channel_length, no_mask_channel_overlap, mask_channel_min_space)
    negative_sampler = _get_conformer_negativer_sampler(extractor_output_dim, encoder_embed_dim, num_negatives, cross_sample_negatives)
    return ConformerWav2Vec2PretrainModel(wav2vec2=wav2vec2, mask_generator=mask_generator, negative_sampler=negative_sampler)