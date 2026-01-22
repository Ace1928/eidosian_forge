import os
import ovs.util
import ovs.vlog
def set_backoff(self, min_backoff, max_backoff):
    """Configures the backoff parameters for this FSM.  'min_backoff' is
        the minimum number of milliseconds, and 'max_backoff' is the maximum,
        between connection attempts.

        'min_backoff' must be at least 1000, and 'max_backoff' must be greater
        than or equal to 'min_backoff'."""
    self.min_backoff = max(min_backoff, 1000)
    if self.max_backoff:
        self.max_backoff = max(max_backoff, 1000)
    else:
        self.max_backoff = 8000
    if self.min_backoff > self.max_backoff:
        self.max_backoff = self.min_backoff
    if self.state == Reconnect.Backoff and self.backoff > self.max_backoff:
        self.backoff = self.max_backoff