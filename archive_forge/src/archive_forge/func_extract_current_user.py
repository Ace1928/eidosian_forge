from __future__ import annotations
from fastapi import Depends, Request
from fastapi.background import BackgroundTasks
from lazyops.libs.fastapi_utils.utils import create_function_wrapper
from ..types import errors
from ..types.current_user import CurrentUser, UserRole
from ..types.security import Authorization, APIKey
from ..utils.lazy import logger
from typing import Optional, List, Annotated, Type, Union
def extract_current_user(*args, _is_optional: Optional[bool]=False, **kwargs) -> CurrentUser:
    """
    Extract The Current User from the Endpoint Function
    """
    user_kw = 'user' if 'user' in kwargs else 'current_user'
    current_user: CurrentUser = kwargs.get(user_kw, None)
    if current_user is None and (not _is_optional):
        raise errors.NoUserException()
    return current_user