import collections
import errno
import heapq
import logging
import math
import os
import pyngus
import select
import socket
import threading
import time
import uuid
def write_socket(self):
    """Called to write to the socket."""
    if self.socket:
        try:
            pyngus.write_socket_output(self.pyngus_conn, self.socket)
            self.pyngus_conn.process(time.monotonic())
        except (socket.timeout, socket.error) as e:
            self.pyngus_conn.close_output()
            self.pyngus_conn.close_input()
            self._handler.socket_error(str(e))