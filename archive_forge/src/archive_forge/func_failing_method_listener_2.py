import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
@on_trait_change('fail')
def failing_method_listener_2(self, name, new):
    self.exceptions_from.append(2)
    raise Exception('error')