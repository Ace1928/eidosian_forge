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
def setup_new_module_schema(module_name: str, roles: List[OpenAPIRoleSpec]):
    """
    Create a new module schema
    """
    global _openapi_schemas_by_role
    if module_name not in _openapi_schemas_by_role:
        _openapi_schemas_by_role[module_name] = {}
        for role in roles:
            _openapi_schemas_by_role[module_name][role.role] = role