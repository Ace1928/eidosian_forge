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
def fill_defaults(self):
    """Fill out default fields in the identity model

        - Ensures all values are defined
        - Fills out derivative values for name fields fields
        - Fills out null values for optional fields
        """
    if not self.username:
        msg = f'user.username must not be empty: {self}'
        raise ValueError(msg)
    if not self.name:
        self.name = self.username
    if not self.display_name:
        self.display_name = self.name