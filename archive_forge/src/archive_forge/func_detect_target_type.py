import contextlib
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
from importlib import import_module
from multiprocessing import get_context
from multiprocessing.context import SpawnProcess
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
import anyio
from .filters import DefaultFilter
from .main import Change, FileChange, awatch, watch
def detect_target_type(target: Union[str, Callable[..., Any]]) -> "Literal['function', 'command']":
    """
    Used by [`run_process`][watchfiles.run_process], [`arun_process`][watchfiles.arun_process]
    and indirectly the CLI to determine the target type with `target_type` is `auto`.

    Detects the target type - either `function` or `command`. This method is only called with `target_type='auto'`.

    The following logic is employed:

    * If `target` is not a string, it is assumed to be a function
    * If `target` ends with `.py` or `.sh`, it is assumed to be a command
    * Otherwise, the target is assumed to be a function if it matches the regex `[a-zA-Z0-9_]+(\\.[a-zA-Z0-9_]+)+`

    If this logic does not work for you, specify the target type explicitly using the `target_type` function argument
    or `--target-type` command line argument.

    Args:
        target: The target value

    Returns:
        either `'function'` or `'command'`
    """
    if not isinstance(target, str):
        return 'function'
    elif target.endswith(('.py', '.sh')):
        return 'command'
    elif re.fullmatch('[a-zA-Z0-9_]+(\\.[a-zA-Z0-9_]+)+', target):
        return 'function'
    else:
        return 'command'