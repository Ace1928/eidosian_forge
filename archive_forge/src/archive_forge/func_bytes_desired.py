from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
def bytes_desired(self):
    """How many bytes can be received without going beyond the next message?

        This is only used with file-descriptor passing, so we don't get too many
        FDs in a single recvmsg call.
        """
    got = self.buf.bytes_buffered
    if got < 16:
        return 16 - got
    if self.next_msg_size is None:
        self.next_msg_size = calc_msg_size(self.buf.peek(16))
    return self.next_msg_size - got