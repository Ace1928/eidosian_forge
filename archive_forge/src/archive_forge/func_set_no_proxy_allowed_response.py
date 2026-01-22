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
def set_no_proxy_allowed_response(self, response):
    fake_response = mock.Mock()
    fake_response.read.return_value = response
    self.opener.return_value.open.return_value = fake_response