from typing import Any, Callable, Dict, Iterator, List, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types, utils
@cli.register_custom_action('Project', (), ('path', 'ref', 'recursive'))
@exc.on_http_error(exc.GitlabGetError)
def repository_tree(self, path: str='', ref: str='', recursive: bool=False, **kwargs: Any) -> Union[gitlab.client.GitlabList, List[Dict[str, Any]]]:
    """Return a list of files in the repository.

        Args:
            path: Path of the top folder (/ by default)
            ref: Reference to a commit or branch
            recursive: Whether to get the tree recursively
            all: If True, return all the items, without pagination
            per_page: Number of items to retrieve per request
            page: ID of the page to return (starts with page 1)
            iterator: If set to True and no pagination option is
                defined, return a generator instead of a list
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server failed to perform the request

        Returns:
            The representation of the tree
        """
    gl_path = f'/projects/{self.encoded_id}/repository/tree'
    query_data: Dict[str, Any] = {'recursive': recursive}
    if path:
        query_data['path'] = path
    if ref:
        query_data['ref'] = ref
    return self.manager.gitlab.http_list(gl_path, query_data=query_data, **kwargs)