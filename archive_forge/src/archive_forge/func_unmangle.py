import importlib
import traceback
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional
from sphinx.ext.autodoc.mock import ismock, undecorate
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import logging
from sphinx.util.inspect import (getannotations, getmro, getslots, isclass, isenumclass,
def unmangle(subject: Any, name: str) -> Optional[str]:
    """Unmangle the given name."""
    try:
        if isclass(subject) and (not name.endswith('__')):
            prefix = '_%s__' % subject.__name__
            if name.startswith(prefix):
                return name.replace(prefix, '__', 1)
            else:
                for cls in subject.__mro__:
                    prefix = '_%s__' % cls.__name__
                    if name.startswith(prefix):
                        return None
    except AttributeError:
        pass
    return name