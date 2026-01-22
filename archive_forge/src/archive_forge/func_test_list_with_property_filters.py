import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_list_with_property_filters(self):
    list(self.mgr.list(filters={'properties': {'ping': 'pong'}}))
    url = f'/v1/images/detail?limit={DEFAULT_PAGE_SIZE}&property-ping=pong'
    expect = [('GET', url, {}, None)]
    self.assertEqual(expect, self.api.calls)