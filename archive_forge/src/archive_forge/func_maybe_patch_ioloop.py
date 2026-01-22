from __future__ import annotations
import json
import os
import sys
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import tornado.httpclient
import tornado.web
from openapi_core import V30RequestValidator, V30ResponseValidator
from openapi_core.spec.paths import Spec
from openapi_core.validation.request.datatypes import RequestParameters
from tornado.httpclient import HTTPRequest, HTTPResponse
from werkzeug.datastructures import Headers, ImmutableMultiDict
from jupyterlab_server.spec import get_openapi_spec
def maybe_patch_ioloop() -> None:
    """a windows 3.8+ patch for the asyncio loop"""
    if sys.platform.startswith('win') and tornado.version_info < (6, 1) and (sys.version_info >= (3, 8)):
        try:
            from asyncio import WindowsProactorEventLoopPolicy, WindowsSelectorEventLoopPolicy
        except ImportError:
            pass
        else:
            from asyncio import get_event_loop_policy, set_event_loop_policy
            if type(get_event_loop_policy()) is WindowsProactorEventLoopPolicy:
                set_event_loop_policy(WindowsSelectorEventLoopPolicy())