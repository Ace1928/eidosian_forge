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
def process_login_form(self, handler: web.RequestHandler) -> User | None:
    """Process login form data

        Return authenticated User if successful, None if not.
        """
    typed_password = handler.get_argument('password', default='')
    new_password = handler.get_argument('new_password', default='')
    user = None
    if not self.auth_enabled:
        self.log.warning('Accepting anonymous login because auth fully disabled!')
        return self.generate_anonymous_user(handler)
    if self.passwd_check(typed_password) and (not new_password):
        return self.generate_anonymous_user(handler)
    elif self.token and self.token == typed_password:
        user = self.generate_anonymous_user(handler)
        if new_password and self.allow_password_change:
            config_dir = handler.settings.get('config_dir', '')
            config_file = os.path.join(config_dir, 'jupyter_server_config.json')
            self.hashed_password = set_password(new_password, config_file=config_file)
            self.log.info(_i18n(f'Wrote hashed password to {config_file}'))
    return user