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
def test_get_my_object_storage_location(self):
    """Test that the my object storage location convert to ''"""
    my_object_storage_locations = [('my-object-storage.com', ''), ('s3-my-object.jp', ''), ('192.168.100.12', '')]
    for url, expected in my_object_storage_locations:
        self._do_test_get_s3_location(url, expected)