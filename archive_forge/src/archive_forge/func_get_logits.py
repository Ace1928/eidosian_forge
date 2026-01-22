import math
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from ..detr import DetrConfig
from .configuration_maskformer import MaskFormerConfig
from .configuration_maskformer_swin import MaskFormerSwinConfig
def get_logits(self, outputs: MaskFormerModelOutput) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    pixel_embeddings = outputs.pixel_decoder_last_hidden_state
    auxiliary_logits: List[str, Tensor] = []
    if self.config.use_auxiliary_loss:
        stacked_transformer_decoder_outputs = torch.stack(outputs.transformer_decoder_hidden_states)
        classes = self.class_predictor(stacked_transformer_decoder_outputs)
        class_queries_logits = classes[-1]
        mask_embeddings = self.mask_embedder(stacked_transformer_decoder_outputs)
        num_embeddings, batch_size, num_queries, num_channels = mask_embeddings.shape
        _, _, height, width = pixel_embeddings.shape
        binaries_masks = torch.zeros((num_embeddings, batch_size, num_queries, height, width), device=mask_embeddings.device)
        for c in range(num_channels):
            binaries_masks += mask_embeddings[..., c][..., None, None] * pixel_embeddings[None, :, None, c]
        masks_queries_logits = binaries_masks[-1]
        for aux_binary_masks, aux_classes in zip(binaries_masks[:-1], classes[:-1]):
            auxiliary_logits.append({'masks_queries_logits': aux_binary_masks, 'class_queries_logits': aux_classes})
    else:
        transformer_decoder_hidden_states = outputs.transformer_decoder_last_hidden_state
        classes = self.class_predictor(transformer_decoder_hidden_states)
        class_queries_logits = classes
        mask_embeddings = self.mask_embedder(transformer_decoder_hidden_states)
        batch_size, num_queries, num_channels = mask_embeddings.shape
        _, _, height, width = pixel_embeddings.shape
        masks_queries_logits = torch.zeros((batch_size, num_queries, height, width), device=mask_embeddings.device)
        for c in range(num_channels):
            masks_queries_logits += mask_embeddings[..., c][..., None, None] * pixel_embeddings[:, None, c]
    return (class_queries_logits, masks_queries_logits, auxiliary_logits)