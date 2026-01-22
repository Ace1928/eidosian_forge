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
def detect_framework(model_path):
    """Detect framework from model_path by analyzing file extensions.

  Args:
    model_path: The local path to the directory that contains the model file.

  Raises:
    PredictionError: If framework can not be identified from model path.

  Returns:
    A string representing the identified framework or None (custom code is
    assumed in this situation).
  """
    num_tensorflow_models = _count_num_files_in_path(model_path, TENSORFLOW_SPECIFIC_MODEL_FILE_NAMES)
    num_xgboost_models = _count_num_files_in_path(model_path, XGBOOST_SPECIFIC_MODEL_FILE_NAMES)
    num_sklearn_models = _count_num_files_in_path(model_path, SCIKIT_LEARN_MODEL_FILE_NAMES)
    num_matches = num_tensorflow_models + num_xgboost_models + num_sklearn_models
    if num_matches > 1:
        error_msg = 'Multiple model files are found in the model_path: {}'.format(model_path)
        logging.critical(error_msg)
        raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, error_msg)
    if num_tensorflow_models == 1:
        return TENSORFLOW_FRAMEWORK_NAME
    elif num_xgboost_models == 1:
        return XGBOOST_FRAMEWORK_NAME
    elif num_sklearn_models == 1:
        model_obj = load_joblib_or_pickle_model(model_path)
        return detect_sk_xgb_framework_from_obj(model_obj)
    else:
        logging.warning('Model files are not found in the model_path.Assumed to be custom code.')
        return None