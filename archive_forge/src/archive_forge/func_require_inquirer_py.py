import os
from argparse import Namespace, _SubParsersAction
from functools import wraps
from tempfile import mkstemp
from typing import Any, Callable, Iterable, List, Optional, Union
from ..utils import CachedRepoInfo, CachedRevisionInfo, HFCacheInfo, scan_cache_dir
from . import BaseHuggingfaceCLICommand
from ._cli_utils import ANSI
def require_inquirer_py(fn: Callable) -> Callable:
    """Decorator to flag methods that require `InquirerPy`."""

    @wraps(fn)
    def _inner(*args, **kwargs):
        if not _inquirer_py_available:
            raise ImportError('The `delete-cache` command requires extra dependencies to work with the TUI.\nPlease run `pip install huggingface_hub[cli]` to install them.\nOtherwise, disable TUI using the `--disable-tui` flag.')
        return fn(*args, **kwargs)
    return _inner