import hashlib
import io
from unittest import mock
import uuid
import boto3
import botocore
from botocore import exceptions as boto_exceptions
from botocore import stub
from oslo_config import cfg
from oslo_utils.secretutils import md5
from oslo_utils import units
import glance_store as store
from glance_store._drivers import s3
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
@mock.patch.object(boto3.session.Session, 'client')
def test_add_different_backend(self, mock_client):
    self.store = s3.Store(self.conf, backend='s3_region2')
    self.store.configure()
    self.register_store_backend_schemes(self.store, 's3', 's3_region2')
    expected_image_id = str(uuid.uuid4())
    expected_s3_size = FIVE_KB
    expected_s3_contents = b'*' * expected_s3_size
    expected_checksum = md5(expected_s3_contents, usedforsecurity=False).hexdigest()
    expected_multihash = hashlib.sha256(expected_s3_contents).hexdigest()
    expected_location = format_s3_location(S3_CONF['s3_store_access_key'], S3_CONF['s3_store_secret_key'], 'http://s3-region2.com', S3_CONF['s3_store_bucket'], expected_image_id)
    image_s3 = io.BytesIO(expected_s3_contents)
    fake_s3_client = botocore.session.get_session().create_client('s3')
    with stub.Stubber(fake_s3_client) as stubber:
        stubber.add_response(method='head_bucket', service_response={}, expected_params={'Bucket': S3_CONF['s3_store_bucket']})
        stubber.add_client_error(method='head_object', service_error_code='404', service_message='', expected_params={'Bucket': S3_CONF['s3_store_bucket'], 'Key': expected_image_id})
        stubber.add_response(method='put_object', service_response={}, expected_params={'Bucket': S3_CONF['s3_store_bucket'], 'Key': expected_image_id, 'Body': botocore.stub.ANY})
        mock_client.return_value = fake_s3_client
        loc, size, checksum, multihash, metadata = self.store.add(expected_image_id, image_s3, expected_s3_size, self.hash_algo)
        self.assertEqual('s3_region2', metadata['store'])
        self.assertEqual(expected_location, loc)
        self.assertEqual(expected_s3_size, size)
        self.assertEqual(expected_checksum, checksum)
        self.assertEqual(expected_multihash, multihash)