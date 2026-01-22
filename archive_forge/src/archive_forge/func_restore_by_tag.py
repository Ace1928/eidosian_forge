from __future__ import annotations
import functools
import numbers
import socket
from bisect import bisect
from collections import namedtuple
from contextlib import contextmanager
from queue import Empty
from time import time
from vine import promise
from kombu.exceptions import InconsistencyError, VersionMismatch
from kombu.log import get_logger
from kombu.utils.compat import register_after_fork
from kombu.utils.encoding import bytes_to_str
from kombu.utils.eventio import ERR, READ, poll
from kombu.utils.functional import accepts_argument
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from kombu.utils.scheduling import cycle_by_name
from kombu.utils.url import _parse_url
from . import virtual
def restore_by_tag(self, tag, client=None, leftmost=False):

    def restore_transaction(pipe):
        p = pipe.hget(self.unacked_key, tag)
        pipe.multi()
        self._remove_from_indices(tag, pipe)
        if p:
            M, EX, RK = loads(bytes_to_str(p))
            self.channel._do_restore_message(M, EX, RK, pipe, leftmost)
    with self.channel.conn_or_acquire(client) as client:
        client.transaction(restore_transaction, self.unacked_key)