from traits.api import HasTraits, TraitError
from traits.testing.unittest_tools import UnittestTools
def test_type_checking(self):
    dummy = self._create_class()
    other_tuple = ('other value', 75, True)
    with self.assertRaises(TraitError):
        dummy.t1 = other_tuple
    self.assertEqual(dummy.t1, VALUES)
    try:
        dummy.t2 = other_tuple
    except TraitError:
        self.fail('Unexpected TraitError when assigning to tuple.')
    self.assertEqual(dummy.t2, other_tuple)