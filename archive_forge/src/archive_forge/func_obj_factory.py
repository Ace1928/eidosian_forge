import threading
import time
import unittest
from traits import trait_notifiers
from traits.api import Callable, Float, HasTraits, on_trait_change
def obj_factory(self):
    return CalledAsDecorator(callback=self.on_foo_notifications)