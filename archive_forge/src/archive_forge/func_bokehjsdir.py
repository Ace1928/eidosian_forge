from __future__ import annotations
import logging # isort:skip
from pathlib import Path
from .deprecation import deprecated
def bokehjsdir(dev: bool=False) -> str:
    """ Get the location of the bokehjs source files.

    By default the files in ``bokeh/server/static`` are used.  If ``dev``
    is ``True``, then the files in ``bokehjs/build`` preferred. However,
    if not available, then a warning is issued and the former files are
    used as a fallback.

    .. note:
        This is a low-level API. Prefer using ``settings.bokehjsdir()``
        instead of this function.

    .. deprecated:: 3.4.0
        Use ``bokehjs_path()`` instead.
    """
    deprecated((3, 4, 0), 'bokehjsdir()', 'bokehjs_path()')
    return str(bokehjs_path(dev))