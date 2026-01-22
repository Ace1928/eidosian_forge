from __future__ import annotations
import re
import json
import copy
import contextlib
import operator
from abc import ABC
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, TYPE_CHECKING
from lazyops.utils import logger
from lazyops.utils.lazy import lazy_import
from lazyops.libs.fastapi_utils.types.user_roles import UserRole
def get_openapi_schema(module_name: str) -> Dict[str, Any]:
    """
    Get the openapi schema
    """
    global _openapi_schemas
    return _openapi_schemas[module_name]