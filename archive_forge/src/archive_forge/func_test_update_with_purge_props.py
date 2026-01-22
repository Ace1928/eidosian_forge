import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_update_with_purge_props(self):
    self.mgr.update('1', purge_props=True)
    expect_headers = {'x-glance-registry-purge-props': 'true'}
    expect = [('PUT', '/v1/images/1', expect_headers, None)]
    self.assertEqual(expect, self.api.calls)