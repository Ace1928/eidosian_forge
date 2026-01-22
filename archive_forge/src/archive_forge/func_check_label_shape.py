import json
import logging
import random
import warnings
import numpy as np
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray._internal import _cvcopyMakeBorder as copyMakeBorder
from .. import io
from .image import RandomOrderAug, ColorJitterAug, LightingAug, ColorNormalizeAug
from .image import ResizeAug, ForceResizeAug, CastAug, HueJitterAug, RandomGrayAug
from .image import fixed_crop, ImageIter, Augmenter
from ..util import is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def check_label_shape(self, label_shape):
    """Checks if the new label shape is valid"""
    if not len(label_shape) == 2:
        raise ValueError('label_shape should have length 2')
    if label_shape[0] < self.label_shape[0]:
        msg = 'Attempts to reduce label count from %d to %d, not allowed.' % (self.label_shape[0], label_shape[0])
        raise ValueError(msg)
    if label_shape[1] != self.provide_label[0][1][2]:
        msg = 'label_shape object width inconsistent: %d vs %d.' % (self.provide_label[0][1][2], label_shape[1])
        raise ValueError(msg)