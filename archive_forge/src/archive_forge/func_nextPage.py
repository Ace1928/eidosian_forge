from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.protocols import basic
from twisted.python.failure import Failure
from twisted.spread import pb
def nextPage(self):
    val = self.string[self.pointer:self.pointer + self.chunkSize]
    self.pointer += self.chunkSize
    if self.pointer >= len(self.string):
        self.stopPaging()
    return val