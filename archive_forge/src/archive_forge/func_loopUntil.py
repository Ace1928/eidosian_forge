import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
def loopUntil(predicate, interval=0):
    """
    Poor excuse for an event notification helper.  This polls a condition and
    calls back a Deferred when it is seen to be true.

    Do not use this function.
    """
    from twisted.internet import task
    d = defer.Deferred()

    def check():
        res = predicate()
        if res:
            d.callback(res)
    call = task.LoopingCall(check)

    def stop(result):
        call.stop()
        return result
    d.addCallback(stop)
    d2 = call.start(interval)
    d2.addErrback(d.errback)
    return d