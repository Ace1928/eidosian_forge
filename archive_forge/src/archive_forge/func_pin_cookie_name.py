from __future__ import annotations
import getpass
import hashlib
import json
import os
import pkgutil
import re
import sys
import time
import typing as t
import uuid
from contextlib import ExitStack
from io import BytesIO
from itertools import chain
from os.path import basename
from os.path import join
from zlib import adler32
from .._internal import _log
from ..exceptions import NotFound
from ..http import parse_cookie
from ..security import gen_salt
from ..utils import send_file
from ..wrappers.request import Request
from ..wrappers.response import Response
from .console import Console
from .tbtools import DebugFrameSummary
from .tbtools import DebugTraceback
from .tbtools import render_console_html
@property
def pin_cookie_name(self) -> str:
    """The name of the pin cookie."""
    if not hasattr(self, '_pin_cookie'):
        pin_cookie = get_pin_and_cookie_name(self.app)
        self._pin, self._pin_cookie = pin_cookie
    return self._pin_cookie