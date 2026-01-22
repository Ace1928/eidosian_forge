from eventlet.patcher import slurp_properties
import sys
import functools
from eventlet import greenthread
from eventlet import patcher
import _thread
def trace_dispatch_return_extend_back(self, frame, t):
    """A hack function to override error checking in parent class.  It
        allows invalid returns (where frames weren't preveiously entered into
        the profiler) which can happen for all the tasklets that suddenly start
        to get monitored. This means that the time will eventually be attributed
        to a call high in the chain, when there is a tasklet switch
        """
    if isinstance(self.cur[-2], Profile.fake_frame):
        return False
        self.trace_dispatch_call(frame, 0)
    return self.trace_dispatch_return(frame, t)