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
def start_process(target: Union[str, Callable[..., Any]], target_type: "Literal['function', 'command']", args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]], changes: Optional[Set[FileChange]]=None) -> 'CombinedProcess':
    if changes is None:
        changes_env_var = '[]'
    else:
        changes_env_var = json.dumps([[c.raw_str(), p] for c, p in changes])
    os.environ['WATCHFILES_CHANGES'] = changes_env_var
    process: 'Union[SpawnProcess, subprocess.Popen[bytes]]'
    if target_type == 'function':
        kwargs = kwargs or {}
        if isinstance(target, str):
            args = (target, get_tty_path(), args, kwargs)
            target_ = run_function
            kwargs = {}
        else:
            target_ = target
        process = spawn_context.Process(target=target_, args=args, kwargs=kwargs)
        process.start()
    else:
        if args or kwargs:
            logger.warning('ignoring args and kwargs for "command" target')
        assert isinstance(target, str), 'target must be a string to run as a command'
        popen_args = split_cmd(target)
        process = subprocess.Popen(popen_args)
    return CombinedProcess(process)