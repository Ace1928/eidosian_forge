from typing import Any, Callable, Dict, Iterator, List, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types, utils
@cli.register_custom_action('Project', (), ('sha', 'format'))
@exc.on_http_error(exc.GitlabListError)
def repository_archive(self, sha: Optional[str]=None, streamed: bool=False, action: Optional[Callable[..., Any]]=None, chunk_size: int=1024, format: Optional[str]=None, path: Optional[str]=None, *, iterator: bool=False, **kwargs: Any) -> Optional[Union[bytes, Iterator[Any]]]:
    """Return an archive of the repository.

        Args:
            sha: ID of the commit (default branch by default)
            streamed: If True the data will be processed by chunks of
                `chunk_size` and each chunk is passed to `action` for
                treatment
            iterator: If True directly return the underlying response
                iterator
            action: Callable responsible of dealing with chunk of
                data
            chunk_size: Size of each chunk
            format: file format (tar.gz by default)
            path: The subpath of the repository to download (all files by default)
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabListError: If the server failed to perform the request

        Returns:
            The binary data of the archive
        """
    url_path = f'/projects/{self.encoded_id}/repository/archive'
    if format:
        url_path += '.' + format
    query_data = {}
    if sha:
        query_data['sha'] = sha
    if path is not None:
        query_data['path'] = path
    result = self.manager.gitlab.http_get(url_path, query_data=query_data, raw=True, streamed=streamed, **kwargs)
    if TYPE_CHECKING:
        assert isinstance(result, requests.Response)
    return utils.response_content(result, streamed, action, chunk_size, iterator=iterator)