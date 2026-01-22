import contextlib
import os
import sys
from typing import List, Optional, Type, Union
from . import i18n, option, osutils, trace
from .lazy_import import lazy_import
import breezy
from breezy import (
from . import errors, registry
from .hooks import Hooks
from .i18n import gettext
from .plugin import disable_plugins, load_plugins, plugin_name
def ignore_pipe(*args, **kwargs):
    try:
        result = func(*args, **kwargs)
        sys.stdout.flush()
        return result
    except OSError as e:
        import errno
        if getattr(e, 'errno', None) is None:
            raise
        if e.errno != errno.EPIPE:
            if sys.platform != 'win32' or e.errno not in (0, errno.EINVAL):
                raise
        pass
    except KeyboardInterrupt:
        pass