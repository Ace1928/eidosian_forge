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
def test_provider_uri(self):
    for prov in ('gs', 's3'):
        uri_str = '%s://' % prov
        uri = boto.storage_uri(uri_str, validate=False, suppress_consec_slashes=False)
        self.assertEqual(prov, uri.scheme)
        self.assertEqual(uri_str, uri.uri)
        self.assertFalse(hasattr(uri, 'versionless_uri'))
        self.assertEqual('', uri.bucket_name)
        self.assertEqual('', uri.object_name)
        self.assertEqual(None, uri.version_id)
        self.assertEqual(None, uri.generation)
        self.assertEqual(uri.names_provider(), True)
        self.assertEqual(uri.names_container(), True)
        self.assertEqual(uri.names_bucket(), False)
        self.assertEqual(uri.names_object(), False)
        self.assertEqual(uri.names_directory(), False)
        self.assertEqual(uri.names_file(), False)
        self.assertEqual(uri.is_stream(), False)
        self.assertEqual(uri.is_version_specific, False)