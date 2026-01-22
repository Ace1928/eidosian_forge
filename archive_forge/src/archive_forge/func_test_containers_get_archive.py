import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_containers_get_archive(self):
    response = self.mgr.get_archive(CONTAINER1['id'], path)
    expect = [('GET', '/v1/containers/%s/get_archive?%s' % (CONTAINER1['id'], parse.urlencode({'path': path})), {'Content-Length': '0'}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(zun_utils.decode_file_data(data), response['data'])