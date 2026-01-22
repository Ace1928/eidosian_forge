from __future__ import annotations
import functools
import inspect
import itertools
import sys
import warnings
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Callable
from xarray.backends.common import BACKEND_ENTRYPOINTS, BackendEntrypoint
from xarray.core.utils import module_available
def guess_engine(store_spec: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore) -> str | type[BackendEntrypoint]:
    engines = list_engines()
    for engine, backend in engines.items():
        try:
            if backend.guess_can_open(store_spec):
                return engine
        except PermissionError:
            raise
        except Exception:
            warnings.warn(f'{engine!r} fails while guessing', RuntimeWarning)
    compatible_engines = []
    for engine, (_, backend_cls) in BACKEND_ENTRYPOINTS.items():
        try:
            backend = backend_cls()
            if backend.guess_can_open(store_spec):
                compatible_engines.append(engine)
        except Exception:
            warnings.warn(f'{engine!r} fails while guessing', RuntimeWarning)
    installed_engines = [k for k in engines if k != 'store']
    if not compatible_engines:
        if installed_engines:
            error_msg = f"did not find a match in any of xarray's currently installed IO backends {installed_engines}. Consider explicitly selecting one of the installed engines via the ``engine`` parameter, or installing additional IO dependencies, see:\nhttps://docs.xarray.dev/en/stable/getting-started-guide/installing.html\nhttps://docs.xarray.dev/en/stable/user-guide/io.html"
        else:
            error_msg = "xarray is unable to open this file because it has no currently installed IO backends. Xarray's read/write support requires installing optional IO dependencies, see:\nhttps://docs.xarray.dev/en/stable/getting-started-guide/installing.html\nhttps://docs.xarray.dev/en/stable/user-guide/io"
    else:
        error_msg = f"found the following matches with the input file in xarray's IO backends: {compatible_engines}. But their dependencies may not be installed, see:\nhttps://docs.xarray.dev/en/stable/user-guide/io.html \nhttps://docs.xarray.dev/en/stable/getting-started-guide/installing.html"
    raise ValueError(error_msg)