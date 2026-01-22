from __future__ import annotations
import binascii
import datetime
import json
import os
import re
import sys
import typing as t
import uuid
from dataclasses import asdict, dataclass
from http.cookies import Morsel
from tornado import escape, httputil, web
from traitlets import Bool, Dict, Type, Unicode, default
from traitlets.config import LoggingConfigurable
from jupyter_server.transutils import _i18n
from .security import passwd_check, set_password
from .utils import get_anonymous_username
def set_login_cookie(self, handler: web.RequestHandler, user: User) -> None:
    """Call this on handlers to set the login cookie for success"""
    cookie_options = {}
    cookie_options.update(self.cookie_options)
    cookie_options.setdefault('httponly', True)
    secure_cookie = self.secure_cookie
    if secure_cookie is None:
        secure_cookie = handler.request.protocol == 'https'
    if secure_cookie:
        cookie_options.setdefault('secure', True)
    cookie_options.setdefault('path', handler.base_url)
    cookie_name = self.get_cookie_name(handler)
    handler.set_secure_cookie(cookie_name, self.user_to_cookie(user), **cookie_options)