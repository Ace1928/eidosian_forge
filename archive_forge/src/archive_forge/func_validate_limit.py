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
@validator('limit', pre=True)
def validate_limit(cls, value):
    from limits import parse
    if value:
        limit = Limit(**value)
        try:
            parse(f'{limit.calls}/{limit.renewal_period}')
        except ValueError:
            raise MlflowException.invalid_parameter_value('Failed to parse the rate limit configuration.Please make sure limit.calls is a positive number andlimit.renewal_period is a right granularity')
    return value