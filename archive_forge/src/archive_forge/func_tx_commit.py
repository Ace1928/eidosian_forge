import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def tx_commit(self):
    """Commit the current transaction.

        This method commits all messages published and acknowledged in
        the current transaction.  A new transaction starts immediately
        after a commit.
        """
    return self.send_method(spec.Tx.Commit, wait=spec.Tx.CommitOk)