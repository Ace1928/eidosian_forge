import sys
from collections import OrderedDict
from ..Qt import QtWidgets
def blockIfUnchanged(func):

    def fn(self, *args, **kwds):
        prevVal = self.value()
        blocked = self.signalsBlocked()
        self.blockSignals(True)
        try:
            ret = func(self, *args, **kwds)
        finally:
            self.blockSignals(blocked)
        if self.value() != prevVal:
            self.currentIndexChanged.emit(self.currentIndex())
        return ret
    return fn