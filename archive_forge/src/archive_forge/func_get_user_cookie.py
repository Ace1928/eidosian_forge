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
def get_user_cookie(self, handler: web.RequestHandler) -> User | None | t.Awaitable[User | None]:
    """Get user from a cookie

        Calls user_from_cookie to deserialize cookie value
        """
    _user_cookie = handler.get_secure_cookie(self.get_cookie_name(handler), **self.get_secure_cookie_kwargs)
    if not _user_cookie:
        return None
    user_cookie = _user_cookie.decode()
    try:
        return self.user_from_cookie(user_cookie)
    except Exception as e:
        self.log.debug(f'Error unpacking user from cookie: cookie={user_cookie}', exc_info=True)
        self.log.error(f'Error unpacking user from cookie: {e}')
        return None