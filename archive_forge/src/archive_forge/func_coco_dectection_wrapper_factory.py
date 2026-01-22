from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
@WRAPPER_FACTORIES.register(datasets.CocoDetection)
def coco_dectection_wrapper_factory(dataset, target_keys):
    target_keys = parse_target_keys(target_keys, available={'segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'boxes', 'masks', 'labels'}, default={'image_id', 'boxes', 'labels'})

    def segmentation_to_mask(segmentation, *, canvas_size):
        from pycocotools import mask
        segmentation = mask.frPyObjects(segmentation, *canvas_size) if isinstance(segmentation, dict) else mask.merge(mask.frPyObjects(segmentation, *canvas_size))
        return torch.from_numpy(mask.decode(segmentation))

    def wrapper(idx, sample):
        image_id = dataset.ids[idx]
        image, target = sample
        if not target:
            return (image, dict(image_id=image_id))
        canvas_size = tuple(F.get_size(image))
        batched_target = list_of_dicts_to_dict_of_lists(target)
        target = {}
        if 'image_id' in target_keys:
            target['image_id'] = image_id
        if 'boxes' in target_keys:
            target['boxes'] = F.convert_bounding_box_format(tv_tensors.BoundingBoxes(batched_target['bbox'], format=tv_tensors.BoundingBoxFormat.XYWH, canvas_size=canvas_size), new_format=tv_tensors.BoundingBoxFormat.XYXY)
        if 'masks' in target_keys:
            target['masks'] = tv_tensors.Mask(torch.stack([segmentation_to_mask(segmentation, canvas_size=canvas_size) for segmentation in batched_target['segmentation']]))
        if 'labels' in target_keys:
            target['labels'] = torch.tensor(batched_target['category_id'])
        for target_key in target_keys - {'image_id', 'boxes', 'masks', 'labels'}:
            target[target_key] = batched_target[target_key]
        return (image, target)
    return wrapper