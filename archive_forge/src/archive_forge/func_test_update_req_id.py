import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_update_req_id(self):
    fields = {'purge_props': True, 'return_req_id': []}
    self.mgr.update('4', **fields)
    expect_headers = {'x-glance-registry-purge-props': 'true'}
    expect = [('PUT', '/v1/images/4', expect_headers, None)]
    self.assertEqual(expect, self.api.calls)
    expect_req_id = ['req-1234']
    self.assertEqual(expect_req_id, fields['return_req_id'])