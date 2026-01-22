import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def function_listener_3(obj, name, new):
    calls_3.append((obj, name, new))