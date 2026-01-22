import unittest
from unittest import mock
from traits.api import HasTraits, Instance, Int, List
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._testing import (
from traits.trait_list_object import TraitList, TraitListObject
def test_trait_list_iter_observables_error(self):
    instance = ClassWithList()
    instance.not_a_trait_list = CustomList()
    observer = ListItemObserver(notify=True, optional=False)
    with self.assertRaises(ValueError) as exception_context:
        next(observer.iter_observables(instance.not_a_trait_list))
    self.assertIn('Expected a TraitList to be observed', str(exception_context.exception))