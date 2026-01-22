from __future__ import annotations
import ast
import sys
import types
from collections.abc import Callable, Iterable
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import cache_from_source, decode_source
from inspect import isclass
from os import PathLike
from types import CodeType, ModuleType, TracebackType
from typing import Sequence, TypeVar
from unittest.mock import patch
from ._config import global_config
from ._transformer import TypeguardTransformer
def install_import_hook(packages: Iterable[str] | None=None, *, cls: type[TypeguardFinder]=TypeguardFinder) -> ImportHookManager:
    """
    Install an import hook that instruments functions for automatic type checking.

    This only affects modules loaded **after** this hook has been installed.

    :param packages: an iterable of package names to instrument, or ``None`` to
        instrument all packages
    :param cls: a custom meta path finder class
    :return: a context manager that uninstalls the hook on exit (or when you call
        ``.uninstall()``)

    .. versionadded:: 2.6

    """
    if packages is None:
        target_packages: list[str] | None = None
    elif isinstance(packages, str):
        target_packages = [packages]
    else:
        target_packages = list(packages)
    for finder in sys.meta_path:
        if isclass(finder) and finder.__name__ == 'PathFinder' and hasattr(finder, 'find_spec'):
            break
    else:
        raise RuntimeError('Cannot find a PathFinder in sys.meta_path')
    hook = cls(target_packages, finder)
    sys.meta_path.insert(0, hook)
    return ImportHookManager(hook)