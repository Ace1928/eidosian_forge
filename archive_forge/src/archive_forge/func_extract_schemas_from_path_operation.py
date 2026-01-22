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
def extract_schemas_from_path_operation(spec: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Extract the schemas from a path operation
    """
    schemas = []
    for response in spec['responses'].values():
        if 'content' not in response:
            continue
        for content in response['content'].values():
            if 'schema' not in content:
                continue
            if 'definitions' in content['schema']:
                schemas += content['schema']['definitions'].keys()
            if 'title' in content['schema']:
                schemas.append(content['schema']['title'])
            if '$ref' in content['schema']:
                schemas.append(content['schema']['$ref'].split('/')[-1])
    return schemas