from tests.compat import mock, unittest
import datetime
import hashlib
import hmac
import locale
import time
import boto.utils
from boto.utils import Password
from boto.utils import pythonize_name
from boto.utils import _build_instance_metadata_url
from boto.utils import get_instance_userdata
from boto.utils import retry_url
from boto.utils import LazyLoadMetadata
from boto.compat import json, _thread
def test_meta_data_with_invalid_json_format_happened_once(self):
    key_data = 'test'
    invalid_data = '{"invalid_json_format" : true,}'
    valid_data = '{ "%s" : {"valid_json_format": true}}' % key_data
    url = '/'.join(['http://169.254.169.254', key_data])
    num_retries = 2
    self.set_normal_response([key_data, invalid_data, valid_data])
    response = LazyLoadMetadata(url, num_retries)
    self.assertEqual(list(response.values())[0], json.loads(valid_data))