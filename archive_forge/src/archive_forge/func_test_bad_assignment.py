import unittest
import uuid
from traits.api import HasTraits, TraitError, UUID
def test_bad_assignment(self):
    with self.assertRaises(TraitError):
        a = A()
        a.id = uuid.uuid4()