import boto
import tempfile
from boto.exception import InvalidUriError
from boto import storage_uri
from boto.compat import urllib
from boto.s3.keyfile import KeyFile
from tests.integration.s3.mock_storage_service import MockBucket
from tests.integration.s3.mock_storage_service import MockBucketStorageUri
from tests.integration.s3.mock_storage_service import MockConnection
from tests.unit import unittest
def test_explicit_file_uri(self):
    tmp_dir = tempfile.tempdir or ''
    uri_str = 'file://%s' % urllib.request.pathname2url(tmp_dir)
    uri = boto.storage_uri(uri_str, validate=False, suppress_consec_slashes=False)
    self.assertEqual('file', uri.scheme)
    self.assertEqual(uri_str, uri.uri)
    self.assertFalse(hasattr(uri, 'versionless_uri'))
    self.assertEqual('', uri.bucket_name)
    self.assertEqual(tmp_dir, uri.object_name)
    self.assertFalse(hasattr(uri, 'version_id'))
    self.assertFalse(hasattr(uri, 'generation'))
    self.assertFalse(hasattr(uri, 'is_version_specific'))
    self.assertEqual(uri.names_provider(), False)
    self.assertEqual(uri.names_bucket(), False)
    self.assertEqual(uri.is_stream(), False)