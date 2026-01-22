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
def input_types_dict(self) -> Dict[str, Union[DataType, np.dtype, Array, Object]]:
    """Maps column names to types, iff this schema declares names."""
    if not self.has_input_names():
        raise MlflowException('Cannot get input types as a dict for schema without names.')
    return {x.name: x.type for x in self.inputs}