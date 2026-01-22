from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
def test_cloudsearch_results_meta(self):
    """Check returned metadata is parsed correctly"""
    search = SearchConnection(endpoint=HOSTNAME)
    results = search.search(q='Test')
    self.assertEqual(results.rank, '-text_relevance')
    self.assertEqual(results.match_expression, 'Test')