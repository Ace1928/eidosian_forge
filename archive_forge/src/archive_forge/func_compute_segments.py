import json
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def compute_segments(mask_probs, pred_scores, pred_labels, mask_threshold: float=0.5, overlap_mask_area_threshold: float=0.8, label_ids_to_fuse: Optional[Set[int]]=None, target_size: Tuple[int, int]=None):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]
    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    segments: List[Dict] = []
    if target_size is not None:
        mask_probs = nn.functional.interpolate(mask_probs.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)[0]
    current_segment_id = 0
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)
    stuff_memory_list: Dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse
        mask_exists, mask_k = check_segment_validity(mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold)
        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1
            segmentation[mask_k] = current_segment_id
            segment_score = round(pred_scores[k].item(), 6)
            segments.append({'id': current_segment_id, 'label_id': pred_class, 'was_fused': should_fuse, 'score': segment_score})
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id
    return (segmentation, segments)