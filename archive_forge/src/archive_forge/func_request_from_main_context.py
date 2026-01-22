import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
def request_from_main_context(self, force_refresh=False):
    """Return the argument passed in to .run_code(for_code)

        Nothing means calls to run_code must be... ???
        """
    if force_refresh:
        value = self.main_context.switch(Refresh())
    else:
        value = self.main_context.switch(Wait())
    if value is SigintHappened:
        raise KeyboardInterrupt()
    return value