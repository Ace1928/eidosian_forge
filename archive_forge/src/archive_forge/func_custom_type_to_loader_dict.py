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
@lru_cache
def custom_type_to_loader_dict():

    def _load_output_parser(config: dict) -> dict:
        """Load output parser."""
        from langchain.schema.output_parser import StrOutputParser
        output_parser_type = config.pop('_type', None)
        if output_parser_type == 'default':
            return StrOutputParser(**config)
        else:
            raise ValueError(f'Unsupported output parser {output_parser_type}')
    return {'default': _load_output_parser}