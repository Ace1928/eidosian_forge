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
def read_socket(self):
    """Called to read from the socket."""
    if self.socket:
        try:
            pyngus.read_socket_input(self.pyngus_conn, self.socket)
            self.pyngus_conn.process(time.monotonic())
        except (socket.timeout, socket.error) as e:
            self.pyngus_conn.close_input()
            self.pyngus_conn.close_output()
            self._handler.socket_error(str(e))