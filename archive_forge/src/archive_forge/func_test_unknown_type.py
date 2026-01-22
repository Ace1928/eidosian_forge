from castellan.common import exception
from castellan.common import objects
from castellan.tests import base
def test_unknown_type(self):
    self.assertRaises(exception.UnknownManagedObjectTypeError, objects.from_dict, {'type': 'non-existing-managed-object-type'})