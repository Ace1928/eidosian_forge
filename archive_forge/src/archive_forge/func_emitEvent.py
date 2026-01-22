import unittest
import inspect
import threading
def emitEvent(self, yowLayerEvent):
    if self.__upper and (not self.__upper.onEvent(yowLayerEvent)):
        if yowLayerEvent.isDetached():
            yowLayerEvent.detached = False
            self.getStack().execDetached(lambda: self.__upper.emitEvent(yowLayerEvent))
        else:
            self.__upper.emitEvent(yowLayerEvent)