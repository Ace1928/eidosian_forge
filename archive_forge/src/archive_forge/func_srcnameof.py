import inspect
import os
import sys
def srcnameof(obj):
    """Returns the most descriptive name of a Python module, class, or function,
    including source information (filename and linenumber), if available.

    Best-effort, but guaranteed to not fail - always returns something.
    """
    name = nameof(obj, quote=True)
    try:
        src_file = inspect.getsourcefile(obj)
    except Exception:
        pass
    else:
        name += f' (file {src_file!r}'
        try:
            _, src_lineno = inspect.getsourcelines(obj)
        except Exception:
            pass
        else:
            name += f', line {src_lineno}'
        name += ')'
    return name