import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
def run_code(self, for_code=None):
    """Returns Truthy values if code finishes, False otherwise

        if for_code is provided, send that value to the code greenlet
        if source code is complete, returns "done"
        if source code is incomplete, returns "unfinished"
        """
    if self.code_context is None:
        assert self.source is not None
        self.code_context = greenlet.greenlet(self._blocking_run_code)
        if is_main_thread():
            self.orig_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self.sigint_handler)
        request = self.code_context.switch()
    else:
        assert self.code_is_waiting
        self.code_is_waiting = False
        if is_main_thread():
            signal.signal(signal.SIGINT, self.sigint_handler)
        if self.sigint_happened_in_main_context:
            self.sigint_happened_in_main_context = False
            request = self.code_context.switch(SigintHappened)
        else:
            request = self.code_context.switch(for_code)
    logger.debug('request received from code was %r', request)
    if not isinstance(request, RequestFromCodeRunner):
        raise ValueError('Not a valid value from code greenlet: %r' % request)
    if isinstance(request, (Wait, Refresh)):
        self.code_is_waiting = True
        if isinstance(request, Refresh):
            self.request_refresh()
        return False
    elif isinstance(request, (Done, Unfinished)):
        self._unload_code()
        if is_main_thread():
            signal.signal(signal.SIGINT, self.orig_sigint_handler)
        self.orig_sigint_handler = None
        return request
    elif isinstance(request, SystemExitRequest):
        self._unload_code()
        raise SystemExitFromCodeRunner(request.args)