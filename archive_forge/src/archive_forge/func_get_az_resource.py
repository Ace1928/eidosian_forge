from __future__ import annotations
from lazyops.libs import lazyload
from lazyops.libs.pooler import ThreadPooler
from lazyops.utils.logs import logger
from lazyops.utils.lazy import lazy_import
from lazyops.utils.helpers import fail_after
from typing import Any, Callable, Dict, List, Optional, Union, Type
def get_az_resource(name: str, *args, **kwargs) -> 'AZResource':
    """
    Returns the AZResource
    """
    schema = get_az_resource_schema(name)
    return schema(*args, **kwargs)