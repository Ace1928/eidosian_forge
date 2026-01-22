import json
import logging
import os
import pathlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pydantic
import yaml
from packaging import version
from packaging.version import Version
from pydantic import ConfigDict, Field, ValidationError, root_validator, validator
from pydantic.json import pydantic_encoder
from mlflow.exceptions import MlflowException
from mlflow.gateway.base_models import ConfigModel, LimitModel, ResponseModel
from mlflow.gateway.constants import (
from mlflow.gateway.utils import (
@validator('model', pre=True)
def validate_model(cls, model):
    if model:
        model_instance = Model(**model)
        if model_instance.provider in Provider.values() and model_instance.config is None:
            raise MlflowException.invalid_parameter_value(f'A config must be supplied when setting a provider. The provider entry for {model_instance.provider} is incorrect.')
    return model