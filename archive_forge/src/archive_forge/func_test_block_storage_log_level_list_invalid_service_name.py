from cinderclient import api_versions
import ddt
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_log_level as service
def test_block_storage_log_level_list_invalid_service_name(self):
    self.volume_client.api_version = api_versions.APIVersion('3.32')
    arglist = ['--host', self.service_log.host, '--service', 'nova-api', '--log-prefix', 'cinder_test.api.common']
    verifylist = [('host', self.service_log.host), ('service', 'nova-api'), ('log_prefix', 'cinder_test.api.common')]
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)