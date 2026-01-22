import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_paginated_list(self):
    images = list(self.mgr.list(page_size=2))
    expect = [('GET', '/v1/images/detail?limit=2', {}, None), ('GET', '/v1/images/detail?limit=2&marker=b', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(3, len(images))
    self.assertEqual('a', images[0].id)
    self.assertEqual('b', images[1].id)
    self.assertEqual('c', images[2].id)