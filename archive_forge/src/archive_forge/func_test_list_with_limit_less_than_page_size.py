import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_list_with_limit_less_than_page_size(self):
    results = list(self.mgr.list(page_size=2, limit=1))
    expect = [('GET', '/v1/images/detail?limit=2', {}, None)]
    self.assertEqual(1, len(results))
    self.assertEqual(expect, self.api.calls)