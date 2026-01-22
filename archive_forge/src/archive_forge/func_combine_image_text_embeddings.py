import collections
import logging
import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import UdopConfig
from transformers.modeling_outputs import (
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ..deprecated._archive_maps import UDOP_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
def combine_image_text_embeddings(image_embeddings, inputs_embeds, bbox, visual_bbox, attention_mask=None, num_patches=14, max_len=0, image_size=224, patch_size=16):
    """
    Combine the image and text embeddings for the input to the encoder/decoder of UDOP.

    First, the image embeddings are created by checking for each visual patch if it is inside the bounding box of a
    token. If it is, the visual patch is combined with the token embedding. Then, the visual bounding boxes are combined
    with the text bounding boxes. Finally, the visual bounding boxes are combined with the text attention mask.
    """
    sequence_length = num_patches
    ocr_points_x = torch.clip(torch.floor((bbox[:, :, 0] + bbox[:, :, 2]) / 2.0 * sequence_length).long(), 0, sequence_length - 1)
    ocr_points_y = torch.clip(torch.floor((bbox[:, :, 1] + bbox[:, :, 3]) / 2.0 * sequence_length).long(), 0, sequence_length - 1) * sequence_length
    ocr_points = ocr_points_x + ocr_points_y
    bbox = bbox.to(torch.float64)
    target_seg = (bbox.mean(-1) == 0.0) | (bbox.mean(-1) == 1.0)
    repeated_vision_embeds = torch.gather(image_embeddings, 1, ocr_points.unsqueeze(-1).repeat(1, 1, image_embeddings.size(-1)))
    repeated_vision_embeds[target_seg] = 0.0
    inputs_embeds += repeated_vision_embeds
    patch_inds = torch.full_like(image_embeddings[:, :, 0], True).bool()
    ind = torch.cat([torch.arange(len(ocr_points))[:, None].repeat(1, ocr_points.size(-1))[:, :, None].to(ocr_points), ocr_points[:, :, None]], dim=-1)
    ind = ind.flatten(0, 1)
    rows, cols = zip(*ind)
    patch_inds[rows, cols] = False
    input_vision_patches = [image_embeddings[i][patch_inds[i]] for i in range(len(patch_inds))]
    if visual_bbox is None:
        visual_bbox = get_visual_bbox(image_size=image_size, patch_size=patch_size)
        visual_bbox = visual_bbox.unsqueeze(0).repeat(image_embeddings.size(0), 1, 1)
        visual_bbox = visual_bbox.to(image_embeddings.device)
    visual_bbox = [visual_bbox[i][patch_inds[i]] for i in range(len(patch_inds))]
    if attention_mask is not None:
        visual_attention_mask = [torch.tensor([1] * len(item)).to(attention_mask) for item in visual_bbox]
    if max_len == 0:
        max_len = image_embeddings.size(1)
    else:
        max_len = max_len - inputs_embeds.size(1)
    inputs_vision_patches = torch.stack([pad_sequence(item, max_len, torch.zeros_like(image_embeddings[0, 0])) for item in input_vision_patches])
    visual_bbox = torch.stack([pad_sequence(item, max_len, torch.zeros_like(bbox[0, 0])) for item in visual_bbox])
    if attention_mask is not None:
        visual_attention_mask = torch.stack([pad_sequence(item, max_len, torch.zeros_like(attention_mask[0, 0])) for item in visual_attention_mask])
    inputs_embeds = torch.cat([inputs_embeds, inputs_vision_patches], 1)
    bbox = torch.cat([bbox, visual_bbox], 1)
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, visual_attention_mask], 1)
    return (inputs_embeds, bbox, attention_mask)