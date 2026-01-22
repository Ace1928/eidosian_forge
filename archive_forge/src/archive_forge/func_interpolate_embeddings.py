import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
import torch
import torch.nn as nn
from ..ops.misc import Conv2dNormActivation, MLP
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
def interpolate_embeddings(image_size: int, patch_size: int, model_state: 'OrderedDict[str, torch.Tensor]', interpolation_mode: str='bicubic', reset_heads: bool=False) -> 'OrderedDict[str, torch.Tensor]':
    """This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    pos_embedding = model_state['encoder.pos_embedding']
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f'Unexpected position embedding shape: {pos_embedding.shape}')
    new_seq_length = (image_size // patch_size) ** 2 + 1
    if new_seq_length != seq_length:
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(f'seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d} and seq_length = {seq_length}')
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size
        new_pos_embedding_img = nn.functional.interpolate(pos_embedding_img, size=new_seq_length_1d, mode=interpolation_mode, align_corners=True)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)
        model_state['encoder.pos_embedding'] = new_pos_embedding
        if reset_heads:
            model_state_copy: 'OrderedDict[str, torch.Tensor]' = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith('heads'):
                    model_state_copy[k] = v
            model_state = model_state_copy
    return model_state