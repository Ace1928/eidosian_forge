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
def new_named_terminal(self, **kwargs: Any) -> tuple[str, PtyWithClients]:
    """Create a new named terminal with an automatic name."""
    name = kwargs['name'] if 'name' in kwargs else self._next_available_name()
    term = self.new_terminal(**kwargs)
    self.log.info('New terminal with automatic name: %s', name)
    term.term_name = name
    self.terminals[name] = term
    self.start_reading(term)
    return (name, term)