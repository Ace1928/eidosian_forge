import weakref
from time import perf_counter
from .functions import SignalBlock
from .Qt import QtCore
from .ThreadsafeTimer import ThreadsafeTimer
def setDelay(self, delay):
    self.delay = delay