import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def obj_collected_callback(weakref):
    obj_collected.append(True)