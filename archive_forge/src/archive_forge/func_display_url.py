import gettext
import io
import sys
import json
import logging
import threading
import tempfile
import os
import shutil
import signal
import socket
import webbrowser
import errno
import random
import jinja2
import tornado.ioloop
import tornado.web
from traitlets.config.application import Application
from traitlets.config.loader import Config
from traitlets import Unicode, Integer, Bool, Dict, List, Callable, default, Type, Bytes
from jupyter_server.services.contents.largefilemanager import LargeFileManager
from jupyter_server.services.kernels.handlers import KernelHandler
from jupyter_server.base.handlers import path_regex, FileFindHandler
from jupyter_server.config_manager import recursive_update
from jupyterlab_server.themes_handler import ThemesHandler
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_core.paths import jupyter_config_path, jupyter_path
from .paths import ROOT, STATIC_ROOT, collect_template_paths, collect_static_paths
from .handler import VoilaHandler
from .treehandler import VoilaTreeHandler
from ._version import __version__
from .static_file_handler import MultiStaticFileHandler, TemplateStaticFileHandler, WhiteListFileHandler
from .configuration import VoilaConfiguration
from .execute import VoilaExecutor
from .exporter import VoilaExporter
from .shutdown_kernel_handler import VoilaShutdownKernelHandler
from .voila_kernel_manager import voila_kernel_manager_factory
from .request_info_handler import RequestInfoSocketHandler
from .utils import create_include_assets_functions
@property
def display_url(self):
    if self.custom_display_url:
        url = self.custom_display_url
        if not url.endswith('/'):
            url += '/'
    else:
        if self.ip in ('', '0.0.0.0'):
            ip = '%s' % socket.gethostname()
        else:
            ip = self.ip
        url = self._url(ip)
    if JUPYTER_SERVER_2 and self.identity_provider.token:
        token = self.identity_provider.token if self.identity_provider.token_generated else '...'
        query = f'?token={token}'
    else:
        query = ''
    return f'{url}{query}'