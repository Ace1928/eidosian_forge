import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def function_listener_2(name, new):
    calls_2.append((name, new))