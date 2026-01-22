import copy
import stevedore
from glance.common import location_strategy
from glance.common.location_strategy import location_order
from glance.common.location_strategy import store_type
from glance.tests.unit import base
def test_load_strategy_modules(self):
    modules = location_strategy._load_strategies()
    self.assertEqual(2, len(modules))
    self.assertEqual(set(['location_order', 'store_type']), set(modules.keys()))
    self.assertEqual(location_strategy._available_strategies, modules)