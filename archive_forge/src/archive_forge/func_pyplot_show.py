from contextlib import contextmanager
import errno
from io import BytesIO
import json
import mimetypes
from pathlib import Path
import random
import sys
import signal
import threading
import tornado.web
import tornado.ioloop
import tornado.websocket
import matplotlib as mpl
from matplotlib.backend_bases import _Backend
from matplotlib._pylab_helpers import Gcf
from . import backend_webagg_core as core
from .backend_webagg_core import (  # noqa: F401 # pylint: disable=W0611
@classmethod
def pyplot_show(cls, *, block=None):
    WebAggApplication.initialize()
    url = 'http://{address}:{port}{prefix}'.format(address=WebAggApplication.address, port=WebAggApplication.port, prefix=WebAggApplication.url_prefix)
    if mpl.rcParams['webagg.open_in_browser']:
        import webbrowser
        if not webbrowser.open(url):
            print(f'To view figure, visit {url}')
    else:
        print(f'To view figure, visit {url}')
    WebAggApplication.start()