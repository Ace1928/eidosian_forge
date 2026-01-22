import asyncio
import os
from typing import Dict
from pathlib import Path
import tornado.web
from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.utils import url_path_join
from nbclient.util import ensure_async
from tornado.httputil import split_host_and_port
from traitlets.traitlets import Bool
from ._version import __version__
from .notebook_renderer import NotebookRenderer
from .request_info_handler import RequestInfoSocketHandler
from .utils import ENV_VARIABLE, create_include_assets_functions
def redirect_to_file(self, path):
    self.redirect(url_path_join(self.base_url, 'voila', 'files', path))