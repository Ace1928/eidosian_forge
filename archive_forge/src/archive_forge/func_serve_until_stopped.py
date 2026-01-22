import errno
import io
import logging
import logging.handlers
import os
import queue
import re
import struct
import threading
import traceback
from socketserver import ThreadingTCPServer, StreamRequestHandler
def serve_until_stopped(self):
    import select
    abort = 0
    while not abort:
        rd, wr, ex = select.select([self.socket.fileno()], [], [], self.timeout)
        if rd:
            self.handle_request()
        logging._acquireLock()
        abort = self.abort
        logging._releaseLock()
    self.server_close()