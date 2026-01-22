from _pydevd_bundle.pydevd_constants import ForkSafeLock, get_global_debugger
import os
import sys
from contextlib import contextmanager
def redirect_stream_to_pydb_io_messages(std):
    """
    :param std:
        'stdout' or 'stderr'
    """
    with _RedirectionsHolder._lock:
        redirect_to_name = '_pydevd_%s_redirect_' % (std,)
        if getattr(_RedirectionsHolder, redirect_to_name) is None:
            wrap_buffer = True
            original = getattr(sys, std)
            redirect_to = RedirectToPyDBIoMessages(1 if std == 'stdout' else 2, original, wrap_buffer)
            start_redirect(keep_original_redirection=True, std=std, redirect_to=redirect_to)
            stack = getattr(_RedirectionsHolder, '_stack_%s' % std)
            setattr(_RedirectionsHolder, redirect_to_name, stack[-1])
            return True
        return False