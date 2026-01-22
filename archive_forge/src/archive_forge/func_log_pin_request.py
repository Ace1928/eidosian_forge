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
def log_pin_request(self) -> Response:
    """Log the pin if needed."""
    if self.pin_logging and self.pin is not None:
        _log('info', ' * To enable the debugger you need to enter the security pin:')
        _log('info', ' * Debugger pin code: %s', self.pin)
    return Response('')