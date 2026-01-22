from typing import Any, cast, Dict, List, Optional, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject, RESTObjectList
from gitlab.mixins import (
from gitlab.types import ArrayAttribute, RequiredOptional
from .custom_attributes import UserCustomAttributeManager  # noqa: F401
from .events import UserEventManager  # noqa: F401
from .personal_access_tokens import UserPersonalAccessTokenManager  # noqa: F401
@cli.register_custom_action('User')
@exc.on_http_error(exc.GitlabUnblockError)
def unblock(self, **kwargs: Any) -> Optional[bool]:
    """Unblock the user.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUnblockError: If the user could not be unblocked

        Returns:
            Whether the user status has been changed
        """
    path = f'/users/{self.encoded_id}/unblock'
    server_data = cast(Optional[bool], self.manager.gitlab.http_post(path, **kwargs))
    if server_data is True:
        self._attrs['state'] = 'active'
    return server_data