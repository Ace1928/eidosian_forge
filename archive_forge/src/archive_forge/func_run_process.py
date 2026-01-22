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
def run_process(*paths: Union[Path, str], target: Union[str, Callable[..., Any]], args: Tuple[Any, ...]=(), kwargs: Optional[Dict[str, Any]]=None, target_type: "Literal['function', 'command', 'auto']"='auto', callback: Optional[Callable[[Set[FileChange]], None]]=None, watch_filter: Optional[Callable[[Change, str], bool]]=DefaultFilter(), grace_period: float=0, debounce: int=1600, step: int=50, debug: bool=False, sigint_timeout: int=5, sigkill_timeout: int=1, recursive: bool=True, ignore_permission_denied: bool=False) -> int:
    """
    Run a process and restart it upon file changes.

    `run_process` can work in two ways:

    * Using `multiprocessing.Process` † to run a python function
    * Or, using `subprocess.Popen` to run a command

    !!! note

        **†** technically `multiprocessing.get_context('spawn').Process` to avoid forking and improve
        code reload/import.

    Internally, `run_process` uses [`watch`][watchfiles.watch] with `raise_interrupt=False` so the function
    exits cleanly upon `Ctrl+C`.

    Args:
        *paths: matches the same argument of [`watch`][watchfiles.watch]
        target: function or command to run
        args: arguments to pass to `target`, only used if `target` is a function
        kwargs: keyword arguments to pass to `target`, only used if `target` is a function
        target_type: type of target. Can be `'function'`, `'command'`, or `'auto'` in which case
            [`detect_target_type`][watchfiles.run.detect_target_type] is used to determine the type.
        callback: function to call on each reload, the function should accept a set of changes as the sole argument
        watch_filter: matches the same argument of [`watch`][watchfiles.watch]
        grace_period: number of seconds after the process is started before watching for changes
        debounce: matches the same argument of [`watch`][watchfiles.watch]
        step: matches the same argument of [`watch`][watchfiles.watch]
        debug: matches the same argument of [`watch`][watchfiles.watch]
        sigint_timeout: the number of seconds to wait after sending sigint before sending sigkill
        sigkill_timeout: the number of seconds to wait after sending sigkill before raising an exception
        recursive: matches the same argument of [`watch`][watchfiles.watch]

    Returns:
        number of times the function was reloaded.

    ```py title="Example of run_process running a function"
    from watchfiles import run_process

    def callback(changes):
        print('changes detected:', changes)

    def foobar(a, b):
        print('foobar called with:', a, b)

    if __name__ == '__main__':
        run_process('./path/to/dir', target=foobar, args=(1, 2), callback=callback)
    ```

    As well as using a `callback` function, changes can be accessed from within the target function,
    using the `WATCHFILES_CHANGES` environment variable.

    ```py title="Example of run_process accessing changes"
    from watchfiles import run_process

    def foobar(a, b, c):
        # changes will be an empty list "[]" the first time the function is called
        changes = os.getenv('WATCHFILES_CHANGES')
        changes = json.loads(changes)
        print('foobar called due to changes:', changes)

    if __name__ == '__main__':
        run_process('./path/to/dir', target=foobar, args=(1, 2, 3))
    ```

    Again with the target as `command`, `WATCHFILES_CHANGES` can be used
    to access changes.

    ```bash title="example.sh"
    echo "changers: ${WATCHFILES_CHANGES}"
    ```

    ```py title="Example of run_process running a command"
    from watchfiles import run_process

    if __name__ == '__main__':
        run_process('.', target='./example.sh')
    ```
    """
    if target_type == 'auto':
        target_type = detect_target_type(target)
    logger.debug('running "%s" as %s', target, target_type)
    catch_sigterm()
    process = start_process(target, target_type, args, kwargs)
    reloads = 0
    if grace_period:
        logger.debug('sleeping for %s seconds before watching for changes', grace_period)
        sleep(grace_period)
    try:
        for changes in watch(*paths, watch_filter=watch_filter, debounce=debounce, step=step, debug=debug, raise_interrupt=False, recursive=recursive, ignore_permission_denied=ignore_permission_denied):
            callback and callback(changes)
            process.stop(sigint_timeout=sigint_timeout, sigkill_timeout=sigkill_timeout)
            process = start_process(target, target_type, args, kwargs, changes)
            reloads += 1
    finally:
        process.stop()
    return reloads