from __future__ import annotations
import asyncio
import codecs
import itertools
import logging
import os
import select
import signal
import warnings
from collections import deque
from concurrent import futures
from typing import TYPE_CHECKING, Any, Coroutine
from tornado.ioloop import IOLoop
def client_disconnected(self, websocket: TermSocket) -> None:
    """Send terminal SIGHUP when client disconnects."""
    self.log.info('Websocket closed, sending SIGHUP to terminal.')
    if websocket.terminal:
        if os.name == 'nt':
            websocket.terminal.kill()
            self.pty_read(websocket.terminal.ptyproc.fd)
            return
        websocket.terminal.killpg(signal.SIGHUP)