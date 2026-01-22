from zope.interface import implementer
from twisted.internet.interfaces import IConsumer, IPushProducer
import pywintypes
import win32api
import win32file
import win32pipe
def bufferFull(self):
    if self.producer is not None:
        self.producerPaused = True
        self.producer.pauseProducing()