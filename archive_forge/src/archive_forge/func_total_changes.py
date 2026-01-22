import asyncio
import logging
import sqlite3
from functools import partial
from pathlib import Path
from queue import Empty, Queue, SimpleQueue
from threading import Thread
from typing import (
from warnings import warn
from .context import contextmanager
from .cursor import Cursor
@property
def total_changes(self) -> int:
    return self._conn.total_changes