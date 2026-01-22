import io
import os
import sys
import zipfile
from anyio.to_thread import run_sync
from jupyter_core.utils import ensure_async
from nbformat import from_dict
from tornado import web
from tornado.log import app_log
from jupyter_server.auth.decorator import authorized
from ..base.handlers import FilesRedirectHandler, JupyterHandler, path_regex
def find_resource_files(output_files_dir):
    """Find the resource files in a directory."""
    files = []
    for dirpath, _, filenames in os.walk(output_files_dir):
        files.extend([os.path.join(dirpath, f) for f in filenames])
    return files