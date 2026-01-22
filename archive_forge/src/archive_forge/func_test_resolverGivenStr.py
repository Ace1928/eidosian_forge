import socket
from queue import Queue
from typing import Callable
from unittest import skipIf
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet._resolver import FirstOneWins
from twisted.internet.base import DelayedCall, ReactorBase, ThreadedResolver
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import IReactorThreads, IReactorTime, IResolverSimple
from twisted.internet.task import Clock
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import SkipTest, TestCase
def test_resolverGivenStr(self):
    """
        L{ThreadedResolver.getHostByName} is passed L{str}, encoded using IDNA
        if required.
        """
    calls = []

    @implementer(IResolverSimple)
    class FakeResolver:

        def getHostByName(self, name, timeouts=()):
            calls.append(name)
            return Deferred()

    class JustEnoughReactor(ReactorBase):

        def installWaker(self):
            pass
    fake = FakeResolver()
    reactor = JustEnoughReactor()
    reactor.installResolver(fake)
    rec = FirstOneWins(Deferred())
    reactor.nameResolver.resolveHostName(rec, 'example.example')
    reactor.nameResolver.resolveHostName(rec, 'example.example')
    reactor.nameResolver.resolveHostName(rec, 'vääntynyt.example')
    reactor.nameResolver.resolveHostName(rec, 'рф.example')
    reactor.nameResolver.resolveHostName(rec, 'xn----7sbb4ac0ad0be6cf.xn--p1ai')
    self.assertEqual(len(calls), 5)
    self.assertEqual(list(map(type, calls)), [str] * 5)
    self.assertEqual('example.example', calls[0])
    self.assertEqual('example.example', calls[1])
    self.assertEqual('xn--vntynyt-5waa.example', calls[2])
    self.assertEqual('xn--p1ai.example', calls[3])
    self.assertEqual('xn----7sbb4ac0ad0be6cf.xn--p1ai', calls[4])