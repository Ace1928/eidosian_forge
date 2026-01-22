import hashlib
import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type, Union, cast
from urllib import parse
import wandb
from wandb import util
from wandb.sdk.lib import hashutil, runid
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia, Media
from .helper_types.bounding_boxes_2d import BoundingBoxes2D
from .helper_types.classes import Classes
from .helper_types.image_mask import ImageMask
@classmethod
def to_uint8(cls, data: 'np.ndarray') -> 'np.ndarray':
    """Convert image data to uint8.

        Convert floating point image on the range [0,1] and integer images on the range
        [0,255] to uint8, clipping if necessary.
        """
    np = util.get_module('numpy', required='wandb.Image requires numpy if not supplying PIL Images: pip install numpy')
    dmin = np.min(data)
    if dmin < 0:
        data = (data - np.min(data)) / np.ptp(data)
    if np.max(data) <= 1.0:
        data = (data * 255).astype(np.int32)
    return data.clip(0, 255).astype(np.uint8)