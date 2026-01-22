import base64
import collections
import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import PredictionError
import six
import tensorflow as tf
def is_single_string_input(self, signature):
    """Returns True if the graph only has one string input tensor."""
    if self.is_single_input(signature):
        dtype = list(signature.inputs.values())[0].dtype
        return dtype == dtypes.string.as_datatype_enum
    return False