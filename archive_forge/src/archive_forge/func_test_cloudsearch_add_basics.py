from tests.unit import unittest
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.document import CommitMismatchError, EncodingError, \
import boto
def test_cloudsearch_add_basics(self):
    """Check that multiple documents are added correctly to AWS"""
    document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
    for key, obj in self.objs.items():
        document.add(key, obj['version'], obj['fields'])
    document.commit()
    args = json.loads(HTTPretty.last_request.body.decode('utf-8'))
    for arg in args:
        self.assertTrue(arg['id'] in self.objs)
        self.assertEqual(arg['version'], self.objs[arg['id']]['version'])
        self.assertEqual(arg['fields']['id'], self.objs[arg['id']]['fields']['id'])
        self.assertEqual(arg['fields']['title'], self.objs[arg['id']]['fields']['title'])
        self.assertEqual(arg['fields']['category'], self.objs[arg['id']]['fields']['category'])