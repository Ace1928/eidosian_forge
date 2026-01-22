import asyncio
import atexit
import base64
import collections
import datetime
import re
import signal
import typing as t
from contextlib import asynccontextmanager, contextmanager
from queue import Empty
from textwrap import dedent
from time import monotonic
from jupyter_client import KernelManager
from jupyter_client.client import KernelClient
from nbformat import NotebookNode
from nbformat.v4 import output_from_msg
from traitlets import Any, Bool, Callable, Dict, Enum, Integer, List, Type, Unicode, default
from traitlets.config.configurable import LoggingConfigurable
from .exceptions import (
from .output_widget import OutputWidget
from .util import ensure_async, run_hook, run_sync
def on_comm_open_jupyter_widget(self, msg: t.Dict) -> t.Optional[t.Any]:
    """Handle a jupyter widget comm open."""
    content = msg['content']
    data = content['data']
    state = data['state']
    comm_id = msg['content']['comm_id']
    module = self.widget_registry.get(state['_model_module'])
    if module:
        widget_class = module.get(state['_model_name'])
        if widget_class:
            return widget_class(comm_id, state, self.kc, self)
    return None