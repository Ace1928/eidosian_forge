import unittest
from traits.util.weakiddict import WeakIDDict, WeakIDKeyDict
def test_weak_id_dict_str_representation(self):
    """ test string representation of the WeakIDDict class. """
    weak_id_dict = WeakIDDict()
    desired_repr = '<WeakIDDict at 0x{0:x}>'.format(id(weak_id_dict))
    self.assertEqual(desired_repr, str(weak_id_dict))
    self.assertEqual(desired_repr, repr(weak_id_dict))