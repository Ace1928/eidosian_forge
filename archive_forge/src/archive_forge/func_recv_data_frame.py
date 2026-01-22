from __future__ import print_function
import socket
import struct
import threading
import time
import six
from ._abnf import *
from ._exceptions import *
from ._handshake import *
from ._http import *
from ._logging import *
from ._socket import *
from ._ssl_compat import *
from ._utils import *
def recv_data_frame(self, control_frame=False):
    """
        Receive data with operation code.

        control_frame: a boolean flag indicating whether to return control frame
        data, defaults to False

        return  value: tuple of operation code and string(byte array) value.
        """
    while True:
        frame = self.recv_frame()
        if not frame:
            raise WebSocketProtocolException('Not a valid frame %s' % frame)
        elif frame.opcode in (ABNF.OPCODE_TEXT, ABNF.OPCODE_BINARY, ABNF.OPCODE_CONT):
            self.cont_frame.validate(frame)
            self.cont_frame.add(frame)
            if self.cont_frame.is_fire(frame):
                return self.cont_frame.extract(frame)
        elif frame.opcode == ABNF.OPCODE_CLOSE:
            self.send_close()
            return (frame.opcode, frame)
        elif frame.opcode == ABNF.OPCODE_PING:
            if len(frame.data) < 126:
                self.pong(frame.data)
            else:
                raise WebSocketProtocolException('Ping message is too long')
            if control_frame:
                return (frame.opcode, frame)
        elif frame.opcode == ABNF.OPCODE_PONG:
            if control_frame:
                return (frame.opcode, frame)