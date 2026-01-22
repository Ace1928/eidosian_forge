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
def remove_gifti_data_array_by_intent(self, intent):
    """Removes all the data arrays with the given intent type"""
    intent2remove = intent_codes.code[intent]
    for dele in self.darrays:
        if dele.intent == intent2remove:
            self.darrays.remove(dele)