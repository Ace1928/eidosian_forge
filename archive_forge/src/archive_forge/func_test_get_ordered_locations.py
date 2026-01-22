import copy
import stevedore
from glance.common import location_strategy
from glance.common.location_strategy import location_order
from glance.common.location_strategy import store_type
from glance.tests.unit import base
def test_get_ordered_locations(self):
    self.config(store_type_preference=['  rbd', ' file', 'swift  ', '  http  ', 'vmware'], group='store_type_location_strategy')
    locs = [{'url': 'file://image0', 'metadata': {'idx': 3}}, {'url': 'rbd://image1', 'metadata': {'idx': 0}}, {'url': 'file://image3', 'metadata': {'idx': 4}}, {'url': 'swift://image4', 'metadata': {'idx': 6}}, {'url': 'cinder://image5', 'metadata': {'idx': 9}}, {'url': 'file://image6', 'metadata': {'idx': 5}}, {'url': 'rbd://image7', 'metadata': {'idx': 1}}, {'url': 'vsphere://image8', 'metadata': {'idx': 8}}]
    ordered_locs = store_type.get_ordered_locations(copy.deepcopy(locs))
    locs.sort(key=lambda loc: loc['metadata']['idx'])
    self.assertEqual(locs, ordered_locs)