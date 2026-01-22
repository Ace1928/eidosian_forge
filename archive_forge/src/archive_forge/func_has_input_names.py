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
def has_input_names(self) -> bool:
    """Return true iff this schema declares names, false otherwise."""
    return self.inputs and self.inputs[0].name is not None