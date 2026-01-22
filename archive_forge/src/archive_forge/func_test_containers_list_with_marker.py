import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_containers_list_with_marker(self):
    expect = [('GET', '/v1/containers/?marker=%s' % CONTAINER2['uuid'], {}, None)]
    self._test_containers_list_with_filters(marker=CONTAINER2['uuid'], expect=expect)