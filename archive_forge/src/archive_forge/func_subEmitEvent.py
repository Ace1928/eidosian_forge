import unittest
import inspect
import threading
def subEmitEvent(self, yowLayerEvent):
    self.onEvent(yowLayerEvent)
    self.emitEvent(yowLayerEvent)