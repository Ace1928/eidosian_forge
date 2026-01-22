import unittest
import inspect
import threading
def emitEventOverrider(self, event):
    self.upperEventSink.append(event)