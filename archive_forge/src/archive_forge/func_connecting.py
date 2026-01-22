import os
import ovs.util
import ovs.vlog
def connecting(self, now):
    """Tell this FSM that a connection or listening attempt is in progress.

        The FSM will start a timer, after which the connection or listening
        attempt will be aborted (by returning ovs.reconnect.DISCONNECT from
        self.run())."""
    if self.state != Reconnect.ConnectInProgress:
        if self.passive:
            self.info_level('%s: listening...' % self.name)
        elif self.backoff < self.max_backoff:
            self.info_level('%s: connecting...' % self.name)
        self._transition(now, Reconnect.ConnectInProgress)