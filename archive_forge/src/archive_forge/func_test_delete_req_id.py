import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_delete_req_id(self):
    params = {'return_req_id': []}
    self.mgr.delete('4', **params)
    expect = [('DELETE', '/v1/images/4', {}, None)]
    self.assertEqual(self.api.calls, expect)
    expect_req_id = ['req-1234']
    self.assertEqual(expect_req_id, params['return_req_id'])