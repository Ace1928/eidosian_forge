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
def is_single_input(self, signature):
    """Returns True if the graph only has one input tensor."""
    return len(signature.inputs) == 1