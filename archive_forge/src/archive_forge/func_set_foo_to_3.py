import threading
import time
import unittest
from traits import trait_notifiers
from traits.api import Callable, Float, HasTraits, on_trait_change
def set_foo_to_3(obj):
    obj.foo = 3