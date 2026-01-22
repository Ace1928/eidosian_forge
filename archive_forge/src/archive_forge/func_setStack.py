import unittest
import inspect
import threading
def setStack(self, stack):
    super(YowParallelLayer, self).setStack(stack)
    for s in self.sublayers:
        s.setStack(self.getStack())