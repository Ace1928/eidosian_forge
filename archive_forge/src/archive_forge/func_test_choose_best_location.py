import copy
import stevedore
from glance.common import location_strategy
from glance.common.location_strategy import location_order
from glance.common.location_strategy import store_type
from glance.tests.unit import base
def test_choose_best_location(self):
    self.config(location_strategy='location_order')
    original_locs = [{'url': 'loc1'}, {'url': 'loc2'}]
    best_loc = location_strategy.choose_best_location(original_locs)
    self.assertNotEqual(id(original_locs), id(best_loc))
    self.assertEqual(original_locs[0], best_loc)