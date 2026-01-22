import builtins
import datetime as dt
import importlib.util
import json
import string
import warnings
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
def read_input(x: dict):
    return TensorSpec.from_json_dict(**x) if x['type'] == 'tensor' else ColSpec.from_json_dict(**x)