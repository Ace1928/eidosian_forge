import copy
from unittest import mock
from unittest.mock import call
import uuid
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import keypair
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
def test_keypair_create_with_key_type(self, sm_mock):
    for key_type in ['x509', 'ssh']:
        self.compute_sdk_client.create_keypair.return_value = self.keypair
        self.data = (self.keypair.created_at, self.keypair.fingerprint, self.keypair.id, self.keypair.is_deleted, self.keypair.name, self.keypair.type, self.keypair.user_id)
        arglist = ['--public-key', self.keypair.public_key, self.keypair.name, '--type', key_type]
        verifylist = [('public_key', self.keypair.public_key), ('name', self.keypair.name), ('type', key_type)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('io.open') as mock_open:
            mock_open.return_value = mock.MagicMock()
            m_file = mock_open.return_value.__enter__.return_value
            m_file.read.return_value = 'dummy'
            columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_keypair.assert_called_with(name=self.keypair.name, public_key=self.keypair.public_key, key_type=key_type)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)