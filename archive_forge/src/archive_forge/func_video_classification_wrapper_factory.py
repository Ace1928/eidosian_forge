from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
def video_classification_wrapper_factory(dataset, target_keys):
    if dataset.video_clips.output_format == 'THWC':
        raise RuntimeError(f"{type(dataset).__name__} with `output_format='THWC'` is not supported by this wrapper, since it is not compatible with the transformations. Please use `output_format='TCHW'` instead.")

    def wrapper(idx, sample):
        video, audio, label = sample
        video = tv_tensors.Video(video)
        return (video, audio, label)
    return wrapper