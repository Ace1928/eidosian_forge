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
def test_get_s3_good_location(self):
    """Test that the s3 location can be derived from the host"""
    good_locations = [('s3.amazonaws.com', ''), ('s3-us-east-1.amazonaws.com', 'us-east-1'), ('s3-us-east-2.amazonaws.com', 'us-east-2'), ('s3-us-west-1.amazonaws.com', 'us-west-1'), ('s3-us-west-2.amazonaws.com', 'us-west-2'), ('s3-ap-east-1.amazonaws.com', 'ap-east-1'), ('s3-ap-south-1.amazonaws.com', 'ap-south-1'), ('s3-ap-northeast-1.amazonaws.com', 'ap-northeast-1'), ('s3-ap-northeast-2.amazonaws.com', 'ap-northeast-2'), ('s3-ap-northeast-3.amazonaws.com', 'ap-northeast-3'), ('s3-ap-southeast-1.amazonaws.com', 'ap-southeast-1'), ('s3-ap-southeast-2.amazonaws.com', 'ap-southeast-2'), ('s3-ca-central-1.amazonaws.com', 'ca-central-1'), ('s3-cn-north-1.amazonaws.com.cn', 'cn-north-1'), ('s3-cn-northwest-1.amazonaws.com.cn', 'cn-northwest-1'), ('s3-eu-central-1.amazonaws.com', 'eu-central-1'), ('s3-eu-west-1.amazonaws.com', 'eu-west-1'), ('s3-eu-west-2.amazonaws.com', 'eu-west-2'), ('s3-eu-west-3.amazonaws.com', 'eu-west-3'), ('s3-eu-north-1.amazonaws.com', 'eu-north-1'), ('s3-sa-east-1.amazonaws.com', 'sa-east-1')]
    for url, expected in good_locations:
        self._do_test_get_s3_location(url, expected)