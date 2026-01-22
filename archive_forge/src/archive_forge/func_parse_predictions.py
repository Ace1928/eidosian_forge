import base64
import collections
import contextlib
import json
import logging
import os
import pickle
import subprocess
import sys
import time
import timeit
from ._interfaces import Model
import six
from tensorflow.python.framework import dtypes  # pylint: disable=g-direct-tensorflow-import
def parse_predictions(response_json):
    """Parses the predictions from the json response from prediction server.

  Args:
    response_json(Text): The JSON formatted response to parse.

  Returns:
    Predictions from the response json.

  Raises:
    ValueError if response_json is malformed.
  """
    if not isinstance(response_json, collections_lib.Mapping):
        raise ValueError('Invalid response received from prediction server: {}'.format(repr(response_json)))
    if PREDICTIONS_KEY not in response_json:
        raise ValueError("Required field '{}' missing in prediction server response: {}".format(PREDICTIONS_KEY, repr(response_json)))
    return response_json.pop(PREDICTIONS_KEY)