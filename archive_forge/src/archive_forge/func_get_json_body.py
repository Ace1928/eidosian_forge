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
def get_json_body(self) -> dict[str, Any] | None:
    """Return the body of the request as JSON data."""
    if not self.request.body:
        return None
    body = self.request.body.strip().decode('utf-8')
    try:
        model = json.loads(body)
    except Exception as e:
        self.log.debug('Bad JSON: %r', body)
        self.log.error("Couldn't parse JSON", exc_info=True)
        raise web.HTTPError(400, 'Invalid JSON in body of request') from e
    return cast('dict[str, Any]', model)