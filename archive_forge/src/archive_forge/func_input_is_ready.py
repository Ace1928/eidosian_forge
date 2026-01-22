from __future__ import unicode_literals
import os
import threading
from prompt_toolkit.utils import is_windows
from .select import select_fds
def input_is_ready(self):
    """
        Return True when the input is ready.
        """
    return self._input_is_ready(wait=False)