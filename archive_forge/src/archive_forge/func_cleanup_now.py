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
def cleanup_now(self):
    """Execute and empty pending cleanup functions immediately.

        After cleanup_now all registered cleanups are forgotten.  add_cleanup
        may be called again after cleanup_now; these cleanups will be called
        after self.run returns or raises (or when cleanup_now is next called).

        This is useful for releasing expensive or contentious resources (such
        as write locks) before doing further work that does not require those
        resources (such as writing results to self.outf). Note though, that
        as it releases all resources, this may release locks that the command
        wants to hold, so use should be done with care.
        """
    self._exit_stack.close()