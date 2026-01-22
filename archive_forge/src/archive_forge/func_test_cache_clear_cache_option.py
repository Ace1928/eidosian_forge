from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from osc_lib import exceptions
from openstackclient.image.v2 import cache
from openstackclient.tests.unit.image.v2 import fakes
def test_cache_clear_cache_option(self):
    arglist = ['--cache']
    verifylist = [('target', 'cache')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.image_client.clear_cache.assert_called_once_with('cache')