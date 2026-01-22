import logging
import warnings
from enum import Enum
from typing import Any, Callable, List, Optional, Union
from ray._private.pydantic_compat import (
from ray._private.utils import import_attr
from ray.serve._private.constants import (
from ray.util.annotations import Deprecated, PublicAPI
@validator('middlewares', always=True)
def warn_for_middlewares(cls, v, values):
    if v:
        warnings.warn('Passing `middlewares` to HTTPOptions is deprecated and will be removed in a future version. Consider using the FastAPI integration to configure middlewares on your deployments: https://docs.ray.io/en/latest/serve/http-guide.html#fastapi-http-deployments')
    return v