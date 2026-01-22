from __future__ import annotations
import asyncio
import datetime as dt
import inspect
import logging
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial, wraps
from typing import (
from urllib.parse import urljoin
from weakref import WeakKeyDictionary
import param
from bokeh.document import Document
from bokeh.document.locking import UnlockedDocumentProxy
from bokeh.io import curdoc as _curdoc
from pyviz_comms import CommManager as _CommManager
from ..util import decode_token, parse_timedelta
from .logging import LOG_SESSION_RENDERED, LOG_USER_MSG
def get_profile(self, profile: str):
    """
        Returns the requested profiling output.

        Arguments
        ---------
        profile: str
          The name of the profiling output to return.

        Returns
        -------
        Profiling output wrapped in a pane.
        """
    from .profile import get_profiles
    return get_profiles({(n, e): ps for (n, e), ps in state._profiles.items() if n == profile})[0][1]