import json
from http import HTTPStatus
from typing import Any, Dict, List
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import allow_unauthenticated, authorized
from jupyter_server.base.handlers import APIHandler, JupyterHandler, path_regex
from jupyter_server.utils import url_escape, url_path_join
def location_url(self, path):
    """Return the full URL location of a file.

        Parameters
        ----------
        path : unicode
            The API path of the file, such as "foo/bar.txt".
        """
    return url_path_join(self.base_url, 'api', 'contents', url_escape(path))