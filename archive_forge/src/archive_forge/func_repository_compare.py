from typing import Any, Callable, Dict, Iterator, List, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types, utils
@cli.register_custom_action('Project', ('from_', 'to'))
@exc.on_http_error(exc.GitlabGetError)
def repository_compare(self, from_: str, to: str, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
    """Return a diff between two branches/commits.

        Args:
            from_: Source branch/SHA
            to: Destination branch/SHA
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server failed to perform the request

        Returns:
            The diff
        """
    path = f'/projects/{self.encoded_id}/repository/compare'
    query_data = {'from': from_, 'to': to}
    return self.manager.gitlab.http_get(path, query_data=query_data, **kwargs)