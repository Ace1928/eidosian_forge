import errno
from zope.interface import implementer
from twisted.internet import error, interfaces, main
from twisted.internet.abstract import _ConsumerMixin, _dataMustBeBytes, _LogOwner
from twisted.internet.iocpreactor import iocpsupport as _iocp
from twisted.internet.iocpreactor.const import ERROR_HANDLE_EOF, ERROR_IO_PENDING
from twisted.python import failure
def doWrite(self):
    if len(self.dataBuffer) - self.offset < self.SEND_LIMIT:
        self.dataBuffer = self.dataBuffer[self.offset:] + b''.join(self._tempDataBuffer)
        self.offset = 0
        self._tempDataBuffer = []
        self._tempDataLen = 0
    evt = _iocp.Event(self._cbWrite, self)
    if self.offset:
        sendView = memoryview(self.dataBuffer)
        evt.buff = buff = sendView[self.offset:]
    else:
        evt.buff = buff = self.dataBuffer
    rc, data = self.writeToHandle(buff, evt)
    if rc and rc != ERROR_IO_PENDING:
        self._handleWrite(rc, data, evt)