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
def register_pipeline(self, task: str, pipeline_class: type, pt_model: Optional[Union[type, Tuple[type]]]=None, tf_model: Optional[Union[type, Tuple[type]]]=None, default: Optional[Dict]=None, type: Optional[str]=None) -> None:
    if task in self.supported_tasks:
        logger.warning(f'{task} is already registered. Overwriting pipeline for task {task}...')
    if pt_model is None:
        pt_model = ()
    elif not isinstance(pt_model, tuple):
        pt_model = (pt_model,)
    if tf_model is None:
        tf_model = ()
    elif not isinstance(tf_model, tuple):
        tf_model = (tf_model,)
    task_impl = {'impl': pipeline_class, 'pt': pt_model, 'tf': tf_model}
    if default is not None:
        if 'model' not in default and ('pt' in default or 'tf' in default):
            default = {'model': default}
        task_impl['default'] = default
    if type is not None:
        task_impl['type'] = type
    self.supported_tasks[task] = task_impl
    pipeline_class._registered_impl = {task: task_impl}