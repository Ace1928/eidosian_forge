import logging
import warnings
from enum import Enum
from typing import Any, Callable, List, Optional, Union
from ray._private.pydantic_compat import (
from ray._private.utils import import_attr
from ray.serve._private.constants import (
from ray.util.annotations import Deprecated, PublicAPI
@validator('location', always=True)
def location_backfill_no_server(cls, v, values):
    if values['host'] is None or v is None:
        return DeploymentMode.NoServer
    return v