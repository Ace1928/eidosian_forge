from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
@WRAPPER_FACTORIES.register(datasets.OxfordIIITPet)
def oxford_iiit_pet_wrapper_factor(dataset, target_keys):

    def wrapper(idx, sample):
        image, target = sample
        if target is not None:
            target = wrap_target_by_type(target, target_types=dataset._target_types, type_wrappers={'segmentation': pil_image_to_mask})
        return (image, target)
    return wrapper