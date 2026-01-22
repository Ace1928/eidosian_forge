from __future__ import annotations
from fastapi import Depends, Request
from fastapi.background import BackgroundTasks
from lazyops.libs.fastapi_utils.utils import create_function_wrapper
from ..types import errors
from ..types.current_user import CurrentUser, UserRole
from ..types.security import Authorization, APIKey
from ..utils.lazy import logger
from typing import Optional, List, Annotated, Type, Union
def require_auth_role(role: Union[UserRole, str, int], enabled: Optional[bool]=True, dry_run: Optional[bool]=False, is_optional: Optional[bool]=False, verbose: Optional[bool]=True):
    """
    Creates an auth role validator wrapper

    Usage:

    >>> @app.get('/admin')
    >>> @require_auth_role(UserRole.ADMIN)
    >>> async def admin(current_user: CurrentUser):
            return {'admin': True}

    """
    if isinstance(role, (str, int)):
        role = UserRole.parse_role(role)

    def validation_func(*args, **kwargs):
        """
        Validation Function
        """
        if not enabled:
            return
        current_user = extract_current_user(*args, _is_optional=is_optional, **kwargs)
        if current_user.role < role:
            if verbose:
                logger.info(f'User {current_user.user_id} has role {current_user.role} which is less than {role}')
            if dry_run:
                return
            raise errors.InvalidAuthRoleException(detail=f'User {current_user.user_id} has role {current_user.role} which is less than or equal to {role}')
        return
    return create_function_wrapper(validation_func)