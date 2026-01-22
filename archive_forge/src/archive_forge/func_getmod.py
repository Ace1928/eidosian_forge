import os
import sys
from types import ModuleType
from .version import version as __version__  # NOQA:F401
def getmod():
    if not mod:
        x = importobj(modpath, None)
        if attrname is not None:
            x = getattr(x, attrname)
        mod.append(x)
    return mod[0]