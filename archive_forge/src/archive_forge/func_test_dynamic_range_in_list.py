import unittest
from traits.api import HasTraits, Int, List, Range, Str, TraitError, Tuple
def test_dynamic_range_in_list(self):

    class HasRangeInList(HasTraits):
        low = Int()
        high = Int()
        digit_sequence = List(Range(low='low', high='high'))
    model = HasRangeInList(low=-1, high=1)
    model.digit_sequence = [-1, 0, 1, 1]
    with self.assertRaises(TraitError):
        model.digit_sequence = [-1, 0, 2, 1]