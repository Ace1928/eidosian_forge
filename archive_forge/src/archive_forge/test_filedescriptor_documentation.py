from zope.interface.verify import verifyClass
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IPushProducer
from twisted.trial.unittest import SynchronousTestCase

        When L{FileDescriptor.writeSomeData} returns C{0} to indicate no more
        data can be written immediately, L{FileDescriptor.doWrite} returns
        L{None}.
        