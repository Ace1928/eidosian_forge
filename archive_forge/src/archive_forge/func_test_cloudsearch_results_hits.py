from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
def test_cloudsearch_results_hits(self):
    """Check that documents are parsed properly from AWS"""
    search = SearchConnection(endpoint=HOSTNAME)
    results = search.search(q='Test')
    hits = list(map(lambda x: x['id'], results.docs))
    self.assertEqual(hits, ['12341', '12342', '12343', '12344', '12345', '12346', '12347'])