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
def new_terminal(self, **kwargs: Any) -> PtyWithClients:
    """Make a new terminal, return a :class:`PtyWithClients` instance."""
    options = self.term_settings.copy()
    options['shell_command'] = self.shell_command
    options.update(kwargs)
    argv = options['shell_command']
    env = self.make_term_env(**options)
    cwd = options.get('cwd', None)
    return PtyWithClients(argv, env, cwd)