from __future__ import annotations
from fastapi import Depends, Request
from fastapi.background import BackgroundTasks
from lazyops.libs.fastapi_utils.utils import create_function_wrapper
from ..types import errors
from ..types.current_user import CurrentUser, UserRole
from ..types.security import Authorization, APIKey
from ..utils.lazy import logger
from typing import Optional, List, Annotated, Type, Union
def auth_role_dependency(role: Union[str, UserRole], disabled: Optional[bool]=None, dry_run: Optional[bool]=False, verbose: Optional[bool]=False):
    """
    Creates an auth role validator wrapper
    """
    user_role = UserRole.parse_role(role) if isinstance(role, str) else role

    async def has_auth_role(current_user: ValidUser):
        """
        Checks if the auth role is valid
        """
        if disabled:
            return
        if current_user.role < user_role:
            if verbose:
                logger.info(f'User {current_user.user_id} does not have required role: {user_role}')
            if dry_run:
                return
            raise errors.InvalidAuthRoleException(detail=f'User {current_user.user_id} does not have required role: {user_role}')
    return Depends(has_auth_role)