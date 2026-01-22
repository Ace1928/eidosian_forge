import collections
import oslotest.base as base
import osc_placement.resources.common as common
def test_url_with_filters_unicode_string(self):
    base_url = '/resource_providers'
    expected = '/resource_providers?name=%D0%BF%D1%80%D0%B8%D0%B2%D0%B5%D1%82'
    actual = common.url_with_filters(base_url, {'name': u'привет'})
    self.assertEqual(expected, actual)