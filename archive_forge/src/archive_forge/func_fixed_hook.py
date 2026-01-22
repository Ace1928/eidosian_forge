from __future__ import annotations
import functools
from pathlib import Path
from typing import Any
from typing import Mapping
import warnings
import pluggy
from ..compat import LEGACY_PATH
from ..compat import legacy_path
from ..deprecated import HOOK_LEGACY_PATH_ARG
@functools.wraps(hook)
def fixed_hook(**kw: Any) -> Any:
    path_value: Path | None = kw.pop(path_var, None)
    fspath_value: LEGACY_PATH | None = kw.pop(fspath_var, None)
    if fspath_value is not None:
        warnings.warn(HOOK_LEGACY_PATH_ARG.format(pylib_path_arg=fspath_var, pathlib_path_arg=path_var), stacklevel=2)
    if path_value is not None:
        if fspath_value is not None:
            _check_path(path_value, fspath_value)
        else:
            fspath_value = legacy_path(path_value)
    else:
        assert fspath_value is not None
        path_value = Path(fspath_value)
    kw[path_var] = path_value
    kw[fspath_var] = fspath_value
    return hook(**kw)