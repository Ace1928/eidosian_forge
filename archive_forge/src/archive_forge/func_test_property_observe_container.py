import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def test_property_observe_container(self):
    instance_observe = ClassWithPropertyObservesItems()
    handler_observe = mock.Mock()
    instance_observe.observe(handler_observe, 'discounted')
    instance_depends_on = ClassWithPropertyDependsOnItems()
    instance_depends_on.on_trait_change(get_otc_handler(mock.Mock()), 'discounted')
    for instance in [instance_observe, instance_depends_on]:
        with self.subTest(instance=instance):
            self.assertEqual(instance.discounted_n_calculations, 0)
            instance.list_of_infos.append(PersonInfo(age=30))
            self.assertEqual(instance.discounted_n_calculations, 1)
            instance.list_of_infos[-1].age = 80
            self.assertEqual(instance.discounted_n_calculations, 2)