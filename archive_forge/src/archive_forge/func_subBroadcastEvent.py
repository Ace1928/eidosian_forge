import unittest
import inspect
import threading
def subBroadcastEvent(self, yowLayerEvent):
    self.onEvent(yowLayerEvent)
    self.broadcastEvent(yowLayerEvent)