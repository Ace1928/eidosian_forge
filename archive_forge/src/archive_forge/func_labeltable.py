from http://www.nitrc.org/projects/gifti/
from __future__ import annotations
import base64
import sys
import warnings
from copy import copy
from typing import Type, cast
import numpy as np
from .. import xmlutils as xml
from ..caret import CaretMetaData
from ..deprecated import deprecate_with_version
from ..filebasedimages import SerializableImage
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from .util import KIND2FMT, array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
from .parse_gifti_fast import GiftiImageParser
@labeltable.setter
def labeltable(self, labeltable):
    """Set the labeltable for this GiftiImage

        Parameters
        ----------
        labeltable : :class:`GiftiLabelTable` instance
        """
    if not isinstance(labeltable, GiftiLabelTable):
        raise TypeError('Not a valid GiftiLabelTable instance')
    self._labeltable = labeltable