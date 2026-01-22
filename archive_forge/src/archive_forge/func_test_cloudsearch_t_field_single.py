from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
def test_cloudsearch_t_field_single(self):
    search = SearchConnection(endpoint=HOSTNAME)
    search.search(q='Test', t={'year': '2001..2007'})
    args = self.get_args(HTTPretty.last_request.raw_requestline)
    self.assertEqual(args[b't-year'], [b'2001..2007'])