from zope.interface import implementer
from twisted.internet.interfaces import IConsumer, IPushProducer
import pywintypes
import win32api
import win32file
import win32pipe
def bufferEmpty(self):
    if self.producer is not None and (not self.streamingProducer or self.producerPaused):
        self.producer.producerPaused = False
        self.producer.resumeProducing()
        return True
    return False