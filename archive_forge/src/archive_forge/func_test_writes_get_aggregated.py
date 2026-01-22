from __future__ import annotations
import gc
from typing import Union
from zope.interface import Interface, directlyProvides, implementer
from zope.interface.verify import verifyObject
from hypothesis import given, strategies as st
from twisted.internet import reactor
from twisted.internet.task import Clock, deferLater
from twisted.python.compat import iterbytes
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol, ServerFactory
from twisted.internet.task import TaskStopped
from twisted.internet.testing import NonStreamingProducer, StringTransport
from twisted.protocols.loopback import collapsingPumpPolicy, loopbackAsync
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SynchronousTestCase, TestCase
@given(st.lists(st.one_of(st.none(), st.integers(min_value=1, max_value=100000).map(lambda length: (b'0123456789ABCDEFGHIJ' * (length // 20 + 1))[:length])), max_size=1000))
def test_writes_get_aggregated(self, writes: list[Union[bytes, None]]) -> None:
    """
        A L{_AggregateSmallWrites} correctly aggregates data for the given
        sequence of writes (indicated by bytes) and increments in the clock
        (indicated by C{None}).

        If multiple writes happen in between reactor iterations, they should
        get written in a batch at the start of the next reactor iteration.
        """
    result: list[bytes] = []
    clock = Clock()
    aggregate = _AggregateSmallWrites(result.append, clock)
    for value in writes:
        if value is None:
            clock.advance(0)
        else:
            aggregate.write(value)
    aggregate.flush()
    self.assertEqual(b''.join(result), b''.join((value for value in writes if value is not None)))
    small_writes = writes[:]
    for chunk in result:
        combined_length = len(chunk)
        while small_writes and small_writes[0] is None:
            small_writes.pop(0)
        small_writes_length = 0
        while small_writes:
            next_original_maybe_write = small_writes.pop(0)
            if next_original_maybe_write is None:
                self.assertEqual(combined_length, small_writes_length)
                break
            else:
                small_writes_length += len(next_original_maybe_write)
                if small_writes_length > aggregate.MAX_BUFFER_SIZE:
                    self.assertEqual(combined_length, small_writes_length)
                    break