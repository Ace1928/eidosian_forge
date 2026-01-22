import urllib.parse
import pytest
from jupyter_server.utils import url_path_join
from jupyterlab_server import LabConfig
from tornado.escape import url_escape
from traitlets import Unicode
from jupyterlab.labapp import LabApp
def client_fetch(*parts, headers=None, params=None, **kwargs):
    path_url = url_escape(url_path_join(*parts), plus=False)
    path_url = url_path_join(jp_base_url, path_url)
    params_url = urllib.parse.urlencode(params or {})
    url = path_url + '?' + params_url
    headers = headers or {}
    headers.update(jp_auth_header)
    return http_server_client.fetch(url, headers=headers, request_timeout=250, **kwargs)