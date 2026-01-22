import errno
import inspect
import json
import logging
import os
import re
import select
import socket
import sys
import time
import traceback
import uuid
from pdb import Pdb
from typing import Callable
import setproctitle
import ray
from ray._private import ray_constants
from ray.experimental.internal_kv import _internal_kv_del, _internal_kv_put
from ray.util.annotations import DeveloperAPI
def listen(self):
    if not self._quiet:
        _cry("RemotePdb session open at %s:%s, use 'ray debug' to connect..." % (self._ip_address, self._listen_socket.getsockname()[1]))
    self._listen_socket.listen(1)
    connection, address = self._listen_socket.accept()
    if not self._quiet:
        _cry('RemotePdb accepted connection from %s.' % repr(address))
    self.handle = _LF2CRLF_FileWrapper(connection)
    Pdb.__init__(self, completekey='tab', stdin=self.handle, stdout=self.handle, skip=['ray.*'])
    self.backup = []
    if self._patch_stdstreams:
        for name in ('stderr', 'stdout', '__stderr__', '__stdout__', 'stdin', '__stdin__'):
            self.backup.append((name, getattr(sys, name)))
            setattr(sys, name, self.handle)
    _RemotePdb.active_instance = self