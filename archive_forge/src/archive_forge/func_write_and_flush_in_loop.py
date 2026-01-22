from __future__ import annotations
import asyncio
import queue
import sys
import threading
import time
from contextlib import contextmanager
from typing import Generator, TextIO, cast
from .application import get_app_session, run_in_terminal
from .output import Output
def write_and_flush_in_loop() -> None:
    run_in_terminal(write_and_flush, in_executor=False)