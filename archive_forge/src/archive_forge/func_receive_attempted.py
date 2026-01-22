import os
import ovs.util
import ovs.vlog
def receive_attempted(self, now):
    """Tell 'fsm' that some attempt to receive data on the connection was
        made at 'now'.  The FSM only allows probe interval timer to expire when
        some attempt to receive data on the connection was received after the
        time when it should have expired.  This helps in the case where there's
        a long delay in the poll loop and then reconnect_run() executes before
        the code to try to receive anything from the remote runs.  (To disable
        this feature, pass None for 'now'.)"""
    self.last_receive_attempt = now