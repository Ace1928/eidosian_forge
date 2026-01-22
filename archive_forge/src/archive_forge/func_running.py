import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
@property
def running(self):
    """Returns greenlet if code has been loaded greenlet has been
        started"""
    return self.source and self.code_context