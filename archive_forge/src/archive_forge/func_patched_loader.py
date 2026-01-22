import contextlib
import importlib
import json
import logging
import os
import re
import shutil
import types
import warnings
from functools import lru_cache
from importlib.util import find_spec
from typing import Callable, NamedTuple
import cloudpickle
import yaml
from packaging import version
from packaging.version import Version
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.utils.class_utils import _get_class_from_string
def patched_loader(*args, **kwargs):
    try:
        return loader_func(*args, **kwargs)
    except ValueError as e:
        if 'This code relies on the pickle module' in str(e):
            raise MlflowException('Since langchain-community 0.0.27, loading a module that relies on the pickle deserialization requires the `allow_dangerous_deserialization` flag to be set to True when loading. However, this flag is not supported by the installed version of LangChain. Please upgrade LangChain to 0.1.14 or above by running `pip install langchain>=0.1.14`.', error_code=INTERNAL_ERROR) from e
        else:
            raise