import sys
import socket
import selectors
from time import monotonic as _time
import warnings
def read_sb_data(self):
    """Return any data available in the SB ... SE queue.

        Return b'' if no SB ... SE available. Should only be called
        after seeing a SB or SE command. When a new SB command is
        found, old unread SB data will be discarded. Don't block.

        """
    buf = self.sbdataq
    self.sbdataq = b''
    return buf