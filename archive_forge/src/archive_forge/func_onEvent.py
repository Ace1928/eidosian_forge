import unittest
import inspect
import threading
def onEvent(self, yowLayerEvent):
    stopEvent = False
    for s in self.sublayers:
        stopEvent = stopEvent or s.onEvent(yowLayerEvent)
    return stopEvent