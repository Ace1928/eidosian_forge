import hashlib
import io
from unittest import mock
import uuid
import boto3
import botocore
from botocore import exceptions as boto_exceptions
from botocore import stub
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import s3
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
@mock.patch.object(boto3.session.Session, 'client')
def test_delete_non_existing(self, mock_client):
    """Test that trying to delete a s3 that doesn't exist raises an error
        """
    bucket, key = ('glance', 'no_exist')
    fake_s3_client = botocore.session.get_session().create_client('s3')
    with stub.Stubber(fake_s3_client) as stubber:
        stubber.add_client_error(method='head_object', service_error_code='404', service_message='\n                                     The specified key does not exist.\n                                     ', expected_params={'Bucket': bucket, 'Key': key})
        fake_s3_client.head_bucket = mock.MagicMock()
        mock_client.return_value = fake_s3_client
        uri = 's3://user:key@auth_address/%s/%s' % (bucket, key)
        loc = location.get_location_from_uri(uri, conf=self.conf)
        self.assertRaises(exceptions.NotFound, self.store.delete, loc)