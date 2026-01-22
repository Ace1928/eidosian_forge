import logging
import warnings
from enum import Enum
from typing import Any, Callable, List, Optional, Union
from ray._private.pydantic_compat import (
from ray._private.utils import import_attr
from ray.serve._private.constants import (
from ray.util.annotations import Deprecated, PublicAPI
@validator('num_cpus', always=True)
def warn_for_num_cpus(cls, v, values):
    if v:
        warnings.warn('Passing `num_cpus` to HTTPOptions is deprecated and will be removed in a future version.')
    return v