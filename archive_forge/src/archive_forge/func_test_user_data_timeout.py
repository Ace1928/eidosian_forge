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
def test_user_data_timeout(self):
    self.set_normal_response(['foo'])
    userdata = get_instance_userdata(timeout=1, num_retries=2)
    self.assertEqual('foo', userdata)
    boto.utils.retry_url.assert_called_with('http://169.254.169.254/latest/user-data', retry_on_404=False, num_retries=2, timeout=1)