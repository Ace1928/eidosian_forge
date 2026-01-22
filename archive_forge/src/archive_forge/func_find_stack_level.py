from __future__ import annotations
import logging # isort:skip
import inspect
import os
import warnings  # lgtm [py/import-and-import-from]
def find_stack_level() -> int:
    """Find the first place in the stack that is not inside Bokeh.

    Inspired by: pandas.util._exceptions.find_stack_level
    """
    import bokeh
    pkg_dir = os.path.dirname(bokeh.__file__)
    frame = inspect.currentframe()
    n = 0
    while frame:
        fname = inspect.getfile(frame)
        if fname.startswith(pkg_dir):
            frame = frame.f_back
            n += 1
        else:
            break
    return n