import asyncio
import contextlib
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from queue import Empty, Queue
from threading import Thread
import numpy as np
import pytest
import requests
from packaging.version import Version
import panel as pn
from panel.io.server import serve
from panel.io.state import state
from panel.pane.alert import Alert
from panel.pane.markup import Markdown
from panel.widgets.button import _ButtonBase
def serve_and_request(app, suffix='', n=1, port=None, **kwargs):
    port = serve_and_wait(app, port=port, **kwargs)
    reqs = [requests.get(f'http://localhost:{port}{suffix}') for i in range(n)]
    return reqs[0] if len(reqs) == 1 else reqs