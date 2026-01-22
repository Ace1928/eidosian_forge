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
def test_convert_dict_to_string(self):
    for case in self.cases:
        self.assertEqual(case[1], parse.unquote_plus(utils.prepare_query_string(case[0])))