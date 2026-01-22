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
@contextmanager
def patch_stdout(raw: bool=False) -> Generator[None, None, None]:
    """
    Replace `sys.stdout` by an :class:`_StdoutProxy` instance.

    Writing to this proxy will make sure that the text appears above the
    prompt, and that it doesn't destroy the output from the renderer.  If no
    application is curring, the behavior should be identical to writing to
    `sys.stdout` directly.

    Warning: If a new event loop is installed using `asyncio.set_event_loop()`,
        then make sure that the context manager is applied after the event loop
        is changed. Printing to stdout will be scheduled in the event loop
        that's active when the context manager is created.

    :param raw: (`bool`) When True, vt100 terminal escape sequences are not
                removed/escaped.
    """
    with StdoutProxy(raw=raw) as proxy:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = cast(TextIO, proxy)
        sys.stderr = cast(TextIO, proxy)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr