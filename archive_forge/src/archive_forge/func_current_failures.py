from typing import Any, cast, Dict, List, TYPE_CHECKING, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
@cli.register_custom_action('GeoNodeManager')
@exc.on_http_error(exc.GitlabGetError)
def current_failures(self, **kwargs: Any) -> List[Dict[str, Any]]:
    """Get the list of failures on the current geo node.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server failed to perform the request

        Returns:
            The list of failures
        """
    result = self.gitlab.http_list('/geo_nodes/current/failures', **kwargs)
    if TYPE_CHECKING:
        assert isinstance(result, list)
    return result