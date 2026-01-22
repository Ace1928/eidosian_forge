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
def test_gs_object_uri_contains_sharp_not_matching_version_syntax(self):
    uri_str = 'gs://bucket/obj#13a990880167400'
    uri = boto.storage_uri(uri_str, validate=False, suppress_consec_slashes=False)
    self.assertEqual('gs', uri.scheme)
    self.assertEqual(uri_str, uri.uri)
    self.assertEqual('gs://bucket/obj#13a990880167400', uri.versionless_uri)
    self.assertEqual('bucket', uri.bucket_name)
    self.assertEqual('obj#13a990880167400', uri.object_name)
    self.assertEqual(None, uri.version_id)
    self.assertEqual(None, uri.generation)
    self.assertEqual(uri.names_provider(), False)
    self.assertEqual(uri.names_container(), False)
    self.assertEqual(uri.names_bucket(), False)
    self.assertEqual(uri.names_object(), True)
    self.assertEqual(uri.names_directory(), False)
    self.assertEqual(uri.names_file(), False)
    self.assertEqual(uri.is_stream(), False)
    self.assertEqual(uri.is_version_specific, False)