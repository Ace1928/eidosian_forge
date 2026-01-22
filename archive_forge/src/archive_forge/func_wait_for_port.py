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
def wait_for_port(stdout):
    nbsr = NBSR(stdout)
    m = None
    output = []
    for _ in range(20):
        o = nbsr.readline(0.5)
        if not o:
            continue
        out = o.decode('utf-8')
        output.append(out)
        m = APP_PATTERN.search(out)
        if m is not None:
            break
    if m is None:
        output = '\n    '.join(output)
        pytest.fail(f'No matching log line in process output, following output was captured:\n\n   {output}')
    return int(m.group(1))