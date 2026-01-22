from __future__ import annotations
import datetime
import errno
import gettext
import hashlib
import hmac
import ipaddress
import json
import logging
import mimetypes
import os
import pathlib
import random
import re
import select
import signal
import socket
import stat
import sys
import threading
import time
import typing as t
import urllib
import warnings
from base64 import encodebytes
from pathlib import Path
import jupyter_client
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.manager import KernelManager
from jupyter_client.session import Session
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from jupyter_core.paths import jupyter_runtime_dir
from jupyter_events.logger import EventLogger
from nbformat.sign import NotebookNotary
from tornado import httpserver, ioloop, web
from tornado.httputil import url_concat
from tornado.log import LogFormatter, access_log, app_log, gen_log
from tornado.netutil import bind_sockets
from tornado.routing import Matcher, Rule
from traitlets import (
from traitlets.config import Config
from traitlets.config.application import boolean_flag, catch_config_error
from jupyter_server import (
from jupyter_server._sysinfo import get_sys_info
from jupyter_server._tz import utcnow
from jupyter_server.auth.authorizer import AllowAllAuthorizer, Authorizer
from jupyter_server.auth.identity import (
from jupyter_server.auth.login import LoginHandler
from jupyter_server.auth.logout import LogoutHandler
from jupyter_server.base.handlers import (
from jupyter_server.extension.config import ExtensionConfigManager
from jupyter_server.extension.manager import ExtensionManager
from jupyter_server.extension.serverextension import ServerExtensionApp
from jupyter_server.gateway.connections import GatewayWebSocketConnection
from jupyter_server.gateway.gateway_client import GatewayClient
from jupyter_server.gateway.managers import (
from jupyter_server.log import log_request
from jupyter_server.services.config import ConfigManager
from jupyter_server.services.contents.filemanager import (
from jupyter_server.services.contents.largefilemanager import AsyncLargeFileManager
from jupyter_server.services.contents.manager import AsyncContentsManager, ContentsManager
from jupyter_server.services.kernels.connection.base import BaseKernelWebsocketConnection
from jupyter_server.services.kernels.connection.channels import ZMQChannelsWebsocketConnection
from jupyter_server.services.kernels.kernelmanager import (
from jupyter_server.services.sessions.sessionmanager import SessionManager
from jupyter_server.utils import (
from jinja2 import Environment, FileSystemLoader
from jupyter_core.paths import secure_write
from jupyter_core.utils import ensure_async
from jupyter_server.transutils import _i18n, trans
from jupyter_server.utils import pathname2url, urljoin
def init_configurables(self) -> None:
    """Initialize configurables."""
    self.gateway_config = GatewayClient.instance(parent=self)
    if not issubclass(self.kernel_manager_class, AsyncMappingKernelManager):
        warnings.warn('The synchronous MappingKernelManager class is deprecated and will not be supported in Jupyter Server 3.0', DeprecationWarning, stacklevel=2)
    if not issubclass(self.contents_manager_class, AsyncContentsManager):
        warnings.warn('The synchronous ContentsManager classes are deprecated and will not be supported in Jupyter Server 3.0', DeprecationWarning, stacklevel=2)
    self.kernel_spec_manager = self.kernel_spec_manager_class(parent=self)
    kwargs = {'parent': self, 'log': self.log, 'connection_dir': self.runtime_dir, 'kernel_spec_manager': self.kernel_spec_manager}
    if jupyter_client.version_info > (8, 3, 0):
        if self.allow_external_kernels:
            external_connection_dir = self.external_connection_dir
            if external_connection_dir is None:
                external_connection_dir = str(Path(self.runtime_dir) / 'external_kernels')
            kwargs['external_connection_dir'] = external_connection_dir
    elif self.allow_external_kernels:
        self.log.warning("Although allow_external_kernels=True, external kernels are not supported because jupyter-client's version does not allow them (should be >8.3.0).")
    self.kernel_manager = self.kernel_manager_class(**kwargs)
    self.contents_manager = self.contents_manager_class(parent=self, log=self.log)
    self.contents_manager.preferred_dir
    self.session_manager = self.session_manager_class(parent=self, log=self.log, kernel_manager=self.kernel_manager, contents_manager=self.contents_manager)
    self.config_manager = self.config_manager_class(parent=self, log=self.log)
    identity_provider_kwargs = {'parent': self, 'log': self.log}
    if self.login_handler_class is not LoginHandler and self.identity_provider_class is PasswordIdentityProvider:
        self.identity_provider_class = LegacyIdentityProvider
        self.log.warning(f'Customizing authentication via ServerApp.login_handler_class={self.login_handler_class} is deprecated in Jupyter Server 2.0. Use ServerApp.identity_provider_class. Falling back on legacy authentication.')
        identity_provider_kwargs['login_handler_class'] = self.login_handler_class
        if self.logout_handler_class:
            identity_provider_kwargs['logout_handler_class'] = self.logout_handler_class
    elif self.login_handler_class is not LoginHandler:
        self.log.warning(f'Ignoring deprecated config ServerApp.login_handler_class={self.login_handler_class}. Superseded by ServerApp.identity_provider_class={{self.identity_provider_class}}.')
    self.identity_provider = self.identity_provider_class(**identity_provider_kwargs)
    if self.identity_provider_class is LegacyIdentityProvider:
        self.tornado_settings['password'] = self.identity_provider.hashed_password
        self.tornado_settings['token'] = self.identity_provider.token
    if self._token_set:
        self.log.warning('ServerApp.token config is deprecated in jupyter-server 2.0. Use IdentityProvider.token')
        if self.identity_provider.token_generated:
            self.identity_provider.token_generated = False
            self.identity_provider.token = self.token
        else:
            self.log.warning('Ignoring deprecated ServerApp.token config')
    self.authorizer = self.authorizer_class(parent=self, log=self.log, identity_provider=self.identity_provider)