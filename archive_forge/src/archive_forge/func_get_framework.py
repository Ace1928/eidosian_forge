import collections
import csv
import importlib
import json
import os
import pickle
import sys
import traceback
import types
import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from contextlib import contextmanager
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from ..dynamic_module_utils import custom_object_save
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..image_processing_utils import BaseImageProcessor
from ..modelcard import ModelCard
from ..models.auto.configuration_auto import AutoConfig
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
def get_framework(model, revision: Optional[str]=None):
    """
    Select framework (TensorFlow or PyTorch) to use.

    Args:
        model (`str`, [`PreTrainedModel`] or [`TFPreTrainedModel`]):
            If both frameworks are installed, picks the one corresponding to the model passed (either a model class or
            the model name). If no specific model is provided, defaults to using PyTorch.
    """
    warnings.warn('`get_framework` is deprecated and will be removed in v5, use `infer_framework_from_model` instead.', FutureWarning)
    if not is_tf_available() and (not is_torch_available()):
        raise RuntimeError('At least one of TensorFlow 2.0 or PyTorch should be installed. To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ To install PyTorch, read the instructions at https://pytorch.org/.')
    if isinstance(model, str):
        if is_torch_available() and (not is_tf_available()):
            model = AutoModel.from_pretrained(model, revision=revision)
        elif is_tf_available() and (not is_torch_available()):
            model = TFAutoModel.from_pretrained(model, revision=revision)
        else:
            try:
                model = AutoModel.from_pretrained(model, revision=revision)
            except OSError:
                model = TFAutoModel.from_pretrained(model, revision=revision)
    framework = infer_framework(model.__class__)
    return framework