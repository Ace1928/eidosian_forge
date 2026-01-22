import copy
import inspect
import io
import re
import warnings
from configparser import (
from dataclasses import dataclass
from pathlib import Path
from types import GeneratorType
from typing import (
import srsly
from .util import SimpleFrozenDict, SimpleFrozenList  # noqa: F401
def copy_model_field(field: ModelField, type_: Any) -> ModelField:
    """Copy a model field and assign a new type, e.g. to accept an Any type
    even though the original value is typed differently.
    """
    return ModelField(name=field.name, type_=type_, class_validators=field.class_validators, model_config=field.model_config, default=field.default, default_factory=field.default_factory, required=field.required)