import errno
from zope.interface import implementer
from twisted.internet import error, interfaces, main
from twisted.internet.abstract import _ConsumerMixin, _dataMustBeBytes, _LogOwner
from twisted.internet.iocpreactor import iocpsupport as _iocp
from twisted.internet.iocpreactor.const import ERROR_HANDLE_EOF, ERROR_IO_PENDING
from twisted.python import failure
def stopConsuming(self):
    """
        Stop consuming data.

        This is called when a producer has lost its connection, to tell the
        consumer to go lose its connection (and break potential circular
        references).
        """
    self.unregisterProducer()
    self.loseConnection()