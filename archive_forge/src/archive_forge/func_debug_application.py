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
def debug_application(self, environ: WSGIEnvironment, start_response: StartResponse) -> t.Iterator[bytes]:
    """Run the application and conserve the traceback frames."""
    contexts: list[t.ContextManager[t.Any]] = []
    if self.evalex:
        environ['werkzeug.debug.preserve_context'] = contexts.append
    app_iter = None
    try:
        app_iter = self.app(environ, start_response)
        yield from app_iter
        if hasattr(app_iter, 'close'):
            app_iter.close()
    except Exception as e:
        if hasattr(app_iter, 'close'):
            app_iter.close()
        tb = DebugTraceback(e, skip=1, hide=not self.show_hidden_frames)
        for frame in tb.all_frames:
            self.frames[id(frame)] = frame
            self.frame_contexts[id(frame)] = contexts
        is_trusted = bool(self.check_pin_trust(environ))
        html = tb.render_debugger_html(evalex=self.evalex, secret=self.secret, evalex_trusted=is_trusted)
        response = Response(html, status=500, mimetype='text/html')
        try:
            yield from response(environ, start_response)
        except Exception:
            environ['wsgi.errors'].write('Debugging middleware caught exception in streamed response at a point where response headers were already sent.\n')
        environ['wsgi.errors'].write(''.join(tb.render_traceback_text()))