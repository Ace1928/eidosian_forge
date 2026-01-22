import unittest
from traits.api import on_trait_change
from traits.adaptation.api import Adapter
@on_trait_change('adaptee', post_init=True)
def check_that_adaptee_start_can_be_accessed(self):
    self.post_init_notifier_called = True