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
def resize_to_smallest(self) -> None:
    """Set the terminal size to that of the smallest client dimensions.

        A terminal not using the full space available is much nicer than a
        terminal trying to use more than the available space, so we keep it
        sized to the smallest client.
        """
    minrows = mincols = 10001
    for client in self.clients:
        rows, cols = client.size
        if rows is not None and rows < minrows:
            minrows = rows
        if cols is not None and cols < mincols:
            mincols = cols
    if minrows == 10001 or mincols == 10001:
        return
    rows, cols = self.ptyproc.getwinsize()
    if (rows, cols) != (minrows, mincols):
        self.ptyproc.setwinsize(minrows, mincols)