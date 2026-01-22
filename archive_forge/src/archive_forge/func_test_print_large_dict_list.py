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
@mock.patch('sys.stdout', io.StringIO())
def test_print_large_dict_list(self):
    dict = {'k': ['foo1', 'bar1', 'foo2', 'bar2', 'foo3', 'bar3', 'foo4', 'bar4']}
    utils.print_dict(dict, wrap=40)
    self.assertEqual('+----------+------------------------------------------+\n| Property | Value                                    |\n+----------+------------------------------------------+\n| k        | ["foo1", "bar1", "foo2", "bar2", "foo3", |\n|          | "bar3", "foo4", "bar4"]                  |\n+----------+------------------------------------------+\n', sys.stdout.getvalue())