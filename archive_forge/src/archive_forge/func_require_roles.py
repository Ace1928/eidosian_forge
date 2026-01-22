from __future__ import annotations
from fastapi import Depends, Request
from fastapi.background import BackgroundTasks
from lazyops.libs.fastapi_utils.utils import create_function_wrapper
from ..types import errors
from ..types.current_user import CurrentUser, UserRole
from ..types.security import Authorization, APIKey
from ..utils.lazy import logger
from typing import Optional, List, Annotated, Type, Union
def require_roles(roles: Union[str, List[str]], require_all: Optional[bool]=False, dry_run: Optional[bool]=False, verbose: Optional[bool]=True):
    """
    Creates a role validator wrapper
    """
    if not isinstance(roles, list):
        roles = [roles]

    def validation_func(*args, **kwargs):
        """
        Validation Function
        """
        current_user = extract_current_user(*args, **kwargs)
        if not current_user.has_user_roles(roles, require_all=require_all):
            if verbose:
                logger.info(f'User {current_user.user_id} does not have required roles: {roles} / {current_user.role}')
            if dry_run:
                return
            raise errors.InvalidRolesException(detail=f'User {current_user.user_id} does not have required roles: {roles}')
    return create_function_wrapper(validation_func)