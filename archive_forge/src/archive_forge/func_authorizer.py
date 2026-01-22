from __future__ import annotations
import functools
import inspect
import ipaddress
import json
import mimetypes
import os
import re
import types
import warnings
from http.client import responses
from logging import Logger
from typing import TYPE_CHECKING, Any, Awaitable, Coroutine, Sequence, cast
from urllib.parse import urlparse
import prometheus_client
from jinja2 import TemplateNotFound
from jupyter_core.paths import is_hidden
from jupyter_events import EventLogger
from tornado import web
from tornado.log import app_log
from traitlets.config import Application
import jupyter_server
from jupyter_server import CallContext
from jupyter_server._sysinfo import get_sys_info
from jupyter_server._tz import utcnow
from jupyter_server.auth.decorator import allow_unauthenticated, authorized
from jupyter_server.auth.identity import User
from jupyter_server.i18n import combine_translations
from jupyter_server.services.security import csp_report_uri
from jupyter_server.utils import (
@property
def authorizer(self) -> Authorizer:
    if 'authorizer' not in self.settings:
        warnings.warn("The Tornado web application does not have an 'authorizer' defined in its settings. In future releases of jupyter_server, this will be a required key for all subclasses of `JupyterHandler`. For an example, see the jupyter_server source code for how to add an authorizer to the tornado settings: https://github.com/jupyter-server/jupyter_server/blob/653740cbad7ce0c8a8752ce83e4d3c2c754b13cb/jupyter_server/serverapp.py#L234-L256", stacklevel=2)
        from jupyter_server.auth import AllowAllAuthorizer
        self.settings['authorizer'] = AllowAllAuthorizer(config=self.settings.get('config', None), identity_provider=self.identity_provider)
    return cast('Authorizer', self.settings.get('authorizer'))