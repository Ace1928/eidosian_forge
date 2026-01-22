from unittest import mock
import testtools
from urllib import parse
from heatclient.common import utils
from heatclient.v1 import resources
def test_list_filtering(self):
    self._test_list(fields={'stack_id': 'teststack', 'filters': {'name': 'rsc_1'}}, expect='/stacks/teststack/resources?%s' % parse.urlencode({'name': 'rsc_1'}, True))