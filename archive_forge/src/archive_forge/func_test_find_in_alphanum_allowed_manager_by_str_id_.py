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
def test_find_in_alphanum_allowed_manager_by_str_id_(self):
    alphanum_manager = FakeManager(True)
    output = utils.find_resource(alphanum_manager, '01234')
    self.assertEqual(output, alphanum_manager.get('01234'))