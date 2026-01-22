import unittest
import inspect
import threading
def broadcastEvent(self, yowLayerEvent):
    if self.__lower and (not self.__lower.onEvent(yowLayerEvent)):
        if yowLayerEvent.isDetached():
            yowLayerEvent.detached = False
            self.getStack().execDetached(lambda: self.__lower.broadcastEvent(yowLayerEvent))
        else:
            self.__lower.broadcastEvent(yowLayerEvent)