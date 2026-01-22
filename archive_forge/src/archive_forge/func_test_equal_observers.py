import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._testing import (
from traits.trait_dict_object import TraitDict
from traits.trait_types import Dict, Str
def test_equal_observers(self):
    observer1 = DictItemObserver(notify=False, optional=False)
    observer2 = DictItemObserver(notify=False, optional=False)
    self.assertEqual(observer1, observer2)
    self.assertEqual(hash(observer1), hash(observer2))