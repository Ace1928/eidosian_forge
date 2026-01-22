import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
def default_fset(self, obj: Any, value: Any) -> None:
    if isinstance(self.path_or_name, str):
        nested_set(obj, self.path_or_name, value)
    elif isinstance(self.path_or_name, list):
        for p, v in zip(self.path_or_name, value):
            nested_set(obj, p, v)
    else:
        raise TypeError(f'Unexpected type for path {type(self.path_or_name)!r}')