import os
import sys
import threading
from . import process
from . import reduction
def get_start_method(self, allow_none=False):
    if self._actual_context is None:
        if allow_none:
            return None
        self._actual_context = self._default_context
    return self._actual_context._name