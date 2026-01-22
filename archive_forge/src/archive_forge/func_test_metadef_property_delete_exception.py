from openstack import exceptions as sdk_exceptions
from osc_lib import exceptions
from openstackclient.image.v2 import metadef_properties
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_metadef_property_delete_exception(self):
    arglist = ['namespace', 'property']
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.image_client.delete_metadef_property.side_effect = sdk_exceptions.ResourceNotFound
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)