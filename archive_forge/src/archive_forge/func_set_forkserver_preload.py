import os
import sys
import threading
from . import process
from . import reduction
def set_forkserver_preload(self, module_names):
    """Set list of module names to try to load in forkserver process.
        This is really just a hint.
        """
    from .forkserver import set_forkserver_preload
    set_forkserver_preload(module_names)