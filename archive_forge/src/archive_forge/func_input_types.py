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
def input_types(self) -> List[Union[DataType, np.dtype, Array, Object]]:
    """Get types for each column in the schema."""
    return [x.type for x in self.inputs]