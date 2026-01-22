import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._testing import (
from traits.trait_dict_object import TraitDict
from traits.trait_types import Dict, Str
def test_trait_dict_iter_observables(self):
    instance = ClassWithDict()
    observer = create_observer(optional=False)
    actual_item, = list(observer.iter_observables(instance.values))
    self.assertIs(actual_item, instance.values)