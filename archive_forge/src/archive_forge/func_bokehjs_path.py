from __future__ import annotations
import logging # isort:skip
from pathlib import Path
from .deprecation import deprecated
def bokehjs_path(dev: bool=False) -> Path:
    """ Get the location of the bokehjs source files.

    By default the files in ``bokeh/server/static`` are used.  If ``dev``
    is ``True``, then the files in ``bokehjs/build`` preferred. However,
    if not available, then a warning is issued and the former files are
    used as a fallback.

    .. note:
        This is a low-level API. Prefer using ``settings.bokehjs_path()``
        instead of this function.
    """
    if dev:
        js_dir = ROOT_DIR.parent.parent / 'bokehjs' / 'build'
        if js_dir.is_dir():
            return js_dir
        else:
            log.warning(f"bokehjs' build directory '{js_dir}' doesn't exist; required by 'settings.dev'")
    return static_path()