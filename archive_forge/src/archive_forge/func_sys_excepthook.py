import sys
import threading
import time
import traceback
from types import SimpleNamespace
def sys_excepthook(self, *args):
    args = SimpleNamespace(exc_type=args[0], exc_value=args[1], exc_traceback=args[2], thread=None)
    return self._excepthook(args, use_thread_hook=False)