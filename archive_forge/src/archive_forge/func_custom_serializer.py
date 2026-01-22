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
def custom_serializer(obj):
    return {'name': obj.name, 'type_': obj.outer_type_, 'class_validators': obj.class_validators, 'model_config': obj.model_config, 'default': obj.default, 'default_factory': obj.default_factory, 'required': obj.required, 'final': obj.final, 'alias': obj.alias, 'field_info': obj.field_info}