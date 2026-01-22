from twisted.internet.interfaces import IReactorThreads, IReactorTime
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.log import msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
def eventSource(reactor, event):
    msg(format='Thread-based event-source scheduling %(event)r', event=event)
    reactor.callFromThread(event)