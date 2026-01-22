import numbers
import os
from typing import TYPE_CHECKING, Optional, Type, Union
import wandb
from wandb import util
from wandb.sdk.lib import runid
from .._private import MEDIA_TMP
from ..base_types.media import Media
Initialize an ImageMask object.

        Arguments:
            val: (dictionary) One of these two keys to represent the image:
                mask_data : (2D numpy array) The mask containing an integer class label
                    for each pixel in the image
                path : (string) The path to a saved image file of the mask
                class_labels : (dictionary of integers to strings, optional) A mapping
                    of the integer class labels in the mask to readable class names.
                    These will default to class_0, class_1, class_2, etc.

        key: (string)
            The readable name or id for this mask type (e.g. predictions, ground_truth)
        