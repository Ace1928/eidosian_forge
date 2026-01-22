from unittest import mock
import ddt
import manilaclient
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient import exceptions
from manilaclient.tests.unit import utils
def test_cli_args_are_copied(self):

    @api_versions.wraps('2.2', '2.6')
    @cliutils.arg('name_1', help='Name of the something')
    @cliutils.arg('action_1', help='Some action')
    def some_func_1(cs, args):
        pass

    @cliutils.arg('name_2', help='Name of the something')
    @cliutils.arg('action_2', help='Some action')
    @api_versions.wraps('2.2', '2.6')
    def some_func_2(cs, args):
        pass
    args_1 = [(('name_1',), {'help': 'Name of the something'}), (('action_1',), {'help': 'Some action'})]
    self.assertEqual(args_1, some_func_1.arguments)
    args_2 = [(('name_2',), {'help': 'Name of the something'}), (('action_2',), {'help': 'Some action'})]
    self.assertEqual(args_2, some_func_2.arguments)