import os
import re
import sys
import typing as t
from pathlib import Path
import zmq
from IPython.core.getipython import get_ipython
from IPython.core.inputtransformer2 import leading_empty_lines
from tornado.locks import Event
from tornado.queues import Queue
from zmq.utils import jsonapi
from .compiler import get_file_name, get_tmp_directory, get_tmp_hash_seed
def connect_tcp_socket(self):
    """Connect to the tcp socket."""
    self.debugpy_stream.socket.connect(self._get_endpoint())
    self.routing_id = self.debugpy_stream.socket.getsockopt(ROUTING_ID)