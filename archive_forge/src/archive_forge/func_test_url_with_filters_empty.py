import collections
import oslotest.base as base
import osc_placement.resources.common as common
def test_url_with_filters_empty(self):
    base_url = '/resource_providers'
    self.assertEqual(base_url, common.url_with_filters(base_url))
    self.assertEqual(base_url, common.url_with_filters(base_url, {}))