from cinderclient import api_versions
import ddt
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_log_level as service
def test_block_storage_log_level_set_invalid_service_name(self):
    self.volume_client.api_version = api_versions.APIVersion('3.32')
    arglist = ['ERROR', '--host', self.service_log.host, '--service', 'nova-api', '--log-prefix', 'cinder.api.common']
    verifylist = [('level', 'ERROR'), ('host', self.service_log.host), ('service', 'nova-api'), ('log_prefix', 'cinder.api.common')]
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)