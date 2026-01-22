import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
@cli.register_custom_action(('Snippet', 'ProjectSnippet', 'ProjectIssue'))
@exc.on_http_error(exc.GitlabGetError)
def user_agent_detail(self, **kwargs: Any) -> Dict[str, Any]:
    """Get the user agent detail.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server cannot perform the request
        """
    path = f'{self.manager.path}/{self.encoded_id}/user_agent_detail'
    result = self.manager.gitlab.http_get(path, **kwargs)
    if TYPE_CHECKING:
        assert not isinstance(result, requests.Response)
    return result