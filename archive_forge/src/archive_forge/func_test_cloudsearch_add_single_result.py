from tests.unit import unittest
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.document import CommitMismatchError, EncodingError, \
import boto
def test_cloudsearch_add_single_result(self):
    """
        Check that the reply from adding a single document is correctly parsed.
        """
    document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
    document.add('1234', 10, {'id': '1234', 'title': 'Title 1', 'category': ['cat_a', 'cat_b', 'cat_c']})
    doc = document.commit()
    self.assertEqual(doc.status, 'success')
    self.assertEqual(doc.adds, 1)
    self.assertEqual(doc.deletes, 0)
    self.assertEqual(doc.doc_service, document)