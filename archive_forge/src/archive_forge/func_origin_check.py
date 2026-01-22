from __future__ import annotations
import typing as t
from jupyter_core.utils import ensure_async
from jupyter_server._tz import utcnow
from jupyter_server.auth.utils import warn_disabled_authorization
from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.base.websocket import WebSocketMixin
from terminado.management import NamedTermManager
from terminado.websocket import TermSocket as BaseTermSocket
from tornado import web
from .base import TerminalsMixin
def origin_check(self, origin: t.Any=None) -> bool:
    """Terminado adds redundant origin_check
        Tornado already calls check_origin, so don't do anything here.
        """
    return True