import os
import queue
import socket
import tempfile
import threading
import types
import uuid
import urllib.parse  # noqa: WPS301
import pytest
import requests
import requests_unixsocket
from pypytools.gc.custom import DefaultGc
from .._compat import bton, ntob
from .._compat import IS_LINUX, IS_MACOS, IS_WINDOWS, SYS_PLATFORM
from ..server import IS_UID_GID_RESOLVABLE, Gateway, HTTPServer
from ..workers.threadpool import ThreadPool
from ..testing import (
@pytest.mark.parametrize(('minthreads', 'maxthreads', 'error'), ((-1, -1, 'min=-1 must be > 0'), (-1, 0, 'min=-1 must be > 0'), (-1, 1, 'min=-1 must be > 0'), (-1, 2, 'min=-1 must be > 0'), (0, -1, 'min=0 must be > 0'), (0, 0, 'min=0 must be > 0'), (0, 1, 'min=0 must be > 0'), (0, 2, 'min=0 must be > 0'), (1, 0, 'Expected an integer or the infinity value for the `max` argument but got 0.'), (1, 0.5, 'Expected an integer or the infinity value for the `max` argument but got 0.5.'), (2, 0, 'Expected an integer or the infinity value for the `max` argument but got 0.'), (2, '1', "Expected an integer or the infinity value for the `max` argument but got '1'."), (2, 1, 'max=1 must be > min=2')))
def test_threadpool_invalid_threadrange(minthreads, maxthreads, error):
    """Test that a ThreadPool rejects invalid min/max values.

    The ThreadPool should raise an error with the proper message when
    initialized with an invalid min+max number of threads.
    """
    with pytest.raises((ValueError, TypeError), match=error):
        ThreadPool(server=None, min=minthreads, max=maxthreads)