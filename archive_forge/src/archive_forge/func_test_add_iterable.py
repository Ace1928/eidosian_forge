import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_add_iterable(self):
    python_set = set()
    iterable = (i for i in range(4))
    python_set.add(iterable)
    ts = TraitSet()
    ts.add(iterable)
    next(iterable)
    self.assertEqual(ts, python_set)