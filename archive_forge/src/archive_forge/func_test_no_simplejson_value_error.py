from boto.compat import json
from tests.compat import mock, unittest
from tests.unit.cloudsearch.test_search import HOSTNAME, \
from boto.cloudsearch.search import SearchConnection, SearchServiceException
def test_no_simplejson_value_error(self):
    with mock.patch.object(json, 'loads', fake_loads_value_error):
        search = SearchConnection(endpoint=HOSTNAME)
        with self.assertRaisesRegexp(SearchServiceException, 'non-json'):
            search.search(q='test')