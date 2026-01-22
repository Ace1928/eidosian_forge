from __future__ import unicode_literals
import struct
from six import int2byte, binary_type, iterbytes
from .log import logger
def command_received(self, command, data):
    if command == DO:
        self.do_received(data)
    elif command == DONT:
        self.dont_received(data)
    elif command == WILL:
        self.will_received(data)
    elif command == WONT:
        self.wont_received(data)
    else:
        logger.info('command received %r %r', command, data)