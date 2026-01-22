from unittest import mock
from openstackclient.common import extension
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_extension_list_volume(self):
    arglist = ['--volume']
    verifylist = [('volume', True)]
    datalist = ((self.volume_extension.name, self.volume_extension.alias, self.volume_extension.description),)
    self._test_extension_list_helper(arglist, verifylist, datalist)
    self.volume_extensions_mock.assert_called_with()