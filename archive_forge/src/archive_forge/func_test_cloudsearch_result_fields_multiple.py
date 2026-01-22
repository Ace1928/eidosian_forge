from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
def test_cloudsearch_result_fields_multiple(self):
    search = SearchConnection(endpoint=HOSTNAME)
    search.search(q='Test', return_fields=['author', 'title'])
    args = self.get_args(HTTPretty.last_request.raw_requestline)
    self.assertEqual(args[b'return-fields'], [b'author,title'])