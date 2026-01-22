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
def get_cookie_name(self, handler: web.RequestHandler) -> str:
    """Return the login cookie name

        Uses IdentityProvider.cookie_name, if defined.
        Default is to generate a string taking host into account to avoid
        collisions for multiple servers on one hostname with different ports.
        """
    if self.cookie_name:
        return self.cookie_name
    else:
        return _non_alphanum.sub('-', f'username-{handler.request.host}')