import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
def http_post(self, path: str, query_data: Optional[Dict[str, Any]]=None, post_data: Optional[Dict[str, Any]]=None, raw: bool=False, files: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
    """Make a POST request to the Gitlab server.

        Args:
            path: Path or full URL to query ('/projects' or
                        'http://whatever/v4/api/projecs')
            query_data: Data to send as query parameters
            post_data: Data to send in the body (will be converted to
                              json by default)
            raw: If True, do not convert post_data to json
            files: The files to send to the server
            **kwargs: Extra options to send to the server (e.g. sudo)

        Returns:
            The parsed json returned by the server if json is return, else the
            raw content

        Raises:
            GitlabHttpError: When the return code is not 2xx
            GitlabParsingError: If the json data could not be parsed
        """
    query_data = query_data or {}
    post_data = post_data or {}
    result = self.http_request('post', path, query_data=query_data, post_data=post_data, files=files, raw=raw, **kwargs)
    content_type = utils.get_content_type(result.headers.get('Content-Type'))
    try:
        if content_type == 'application/json':
            json_result = result.json()
            if TYPE_CHECKING:
                assert isinstance(json_result, dict)
            return json_result
    except Exception as e:
        raise gitlab.exceptions.GitlabParsingError(error_message='Failed to parse the server message') from e
    return result