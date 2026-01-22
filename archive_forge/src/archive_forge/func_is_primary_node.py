from __future__ import annotations
import os
import contextlib
from enum import Enum
from pathlib import Path
from pydantic import model_validator
from lazyops.utils.logs import Logger, null_logger
from lazyops.imports._pydantic import BaseAppSettings, BaseModel
from lazyops.libs.abcs.state import GlobalContext
from lazyops.libs.fastapi_utils.types.persistence import TemporaryData
from typing import List, Optional, Dict, Any, Callable, Union, Type, TYPE_CHECKING
from .types import AppEnv, get_app_env
@property
def is_primary_node(self) -> bool:
    """
        Returns whether or not this is the primary node
        """
    return self.host_name[-1] == '0' if self.in_k8s else True