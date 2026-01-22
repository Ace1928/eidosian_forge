import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
def test_slice_deletion_bad_length_computation(self):

    class IHasConstrainedList(HasTraits):
        foo = List(Str, minlen=3)
    f = IHasConstrainedList(foo=['zero', 'one', 'two', 'three'])
    with self.assertRaises(TraitError):
        del f.foo[::3]