from tests.unit import unittest
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.document import CommitMismatchError, EncodingError, \
import boto
def test_cloudsearch_add_results(self):
    """
        Check that the result from adding multiple documents is parsed
        correctly.
        """
    document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
    for key, obj in self.objs.items():
        document.add(key, obj['version'], obj['fields'])
    doc = document.commit()
    self.assertEqual(doc.status, 'success')
    self.assertEqual(doc.adds, len(self.objs))
    self.assertEqual(doc.deletes, 0)