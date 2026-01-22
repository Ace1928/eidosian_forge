from __future__ import annotations
import os.path
import sys
import time
import typing as t
from pstats import Stats
def runapp() -> None:
    app_iter = self._app(environ, t.cast('StartResponse', catching_start_response))
    response_body.extend(app_iter)
    if hasattr(app_iter, 'close'):
        app_iter.close()