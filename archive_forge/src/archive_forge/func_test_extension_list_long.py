from unittest import mock
from openstackclient.common import extension
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_extension_list_long(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    datalist = ((self.identity_extension.name, self.identity_extension.alias, self.identity_extension.description, self.identity_extension.namespace, '', self.identity_extension.links), (self.compute_extension.name, self.compute_extension.alias, self.compute_extension.description, self.compute_extension.namespace, self.compute_extension.updated_at, self.compute_extension.links), (self.volume_extension.name, self.volume_extension.alias, self.volume_extension.description, '', self.volume_extension.updated_at, self.volume_extension.links), (self.network_extension.name, self.network_extension.alias, self.network_extension.description, '', self.network_extension.updated_at, self.network_extension.links))
    self._test_extension_list_helper(arglist, verifylist, datalist, True)
    self.identity_extensions_mock.list.assert_called_with()
    self.compute_extensions_mock.assert_called_with()
    self.volume_extensions_mock.assert_called_with()
    self.network_client.extensions.assert_called_with()