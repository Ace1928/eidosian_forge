import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._testing import (
from traits.trait_dict_object import TraitDict
from traits.trait_types import Dict, Str
def test_iter_objects_from_custom_trait_dict(self):
    observer = create_observer(optional=False)
    custom_trait_dict = CustomTraitDict({'1': 1, '2': 2})
    actual = list(observer.iter_objects(custom_trait_dict))
    self.assertCountEqual(actual, [1, 2])