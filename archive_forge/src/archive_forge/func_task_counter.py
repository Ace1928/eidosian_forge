import asyncio
import os
from typing import Awaitable, Tuple, Type, TypeVar, Union
from typing import Dict as TypeDict
from typing import List as TypeList
from pathlib import Path
from traitlets.traitlets import Dict, Float, List, default
from nbclient.util import ensure_async
import re
from .notebook_renderer import NotebookRenderer
from .utils import ENV_VARIABLE
def task_counter(tk):
    nonlocal heated
    heated += 1
    if heated == kernel_size:
        self.log.info('Kernel pool of %s is filled with %s kernel(s)', notebook_name, kernel_size)