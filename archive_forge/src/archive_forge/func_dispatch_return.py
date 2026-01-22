import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def dispatch_return(self, frame, arg):
    """Invoke user function and return trace function for return event.

        If the debugger stops on this function return, invoke
        self.user_return(). Raise BdbQuit if self.quitting is set.
        Return self.trace_dispatch to continue tracing in this scope.
        """
    if self.stop_here(frame) or frame == self.returnframe:
        if self.stopframe and frame.f_code.co_flags & GENERATOR_AND_COROUTINE_FLAGS:
            return self.trace_dispatch
        try:
            self.frame_returning = frame
            self.user_return(frame, arg)
        finally:
            self.frame_returning = None
        if self.quitting:
            raise BdbQuit
        if self.stopframe is frame and self.stoplineno != -1:
            self._set_stopinfo(None, None)
    return self.trace_dispatch