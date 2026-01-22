import unittest
from traits.api import HasTraits, Instance, Str
def set_shared_copy(self, value):
    """ Change the copy style for the 'shared' traits. """
    self.foo.base_trait('shared').copy = value
    self.bar.base_trait('shared').copy = value
    self.baz.base_trait('shared').copy = value