import io
import sys
from unittest import mock
from urllib import parse
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils as test_utils
from novaclient import utils
from novaclient.v2 import servers
def test_find_by_str_lower_name_mixed(self):
    output = utils.find_resource(self.manager, 'mixed')
    self.assertEqual(output, self.manager.get('12345678'))