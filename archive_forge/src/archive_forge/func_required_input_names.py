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
def required_input_names(self) -> List[Union[str, int]]:
    """Get list of required data names or range of indices if schema has no names."""
    return [x.name or i for i, x in enumerate(self.inputs) if x.required]