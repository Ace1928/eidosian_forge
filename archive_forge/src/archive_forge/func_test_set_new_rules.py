import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import mapping
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_set_new_rules(self):
    arglist = ['--rules', identity_fakes.mapping_rules_file_path, identity_fakes.mapping_id]
    verifylist = [('mapping', identity_fakes.mapping_id), ('rules', identity_fakes.mapping_rules_file_path)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    mocker = mock.Mock()
    mocker.return_value = identity_fakes.MAPPING_RULES_2
    with mock.patch('openstackclient.identity.v3.mapping.SetMapping._read_rules', mocker):
        result = self.cmd.take_action(parsed_args)
    self.mapping_mock.update.assert_called_with(mapping=identity_fakes.mapping_id, rules=identity_fakes.MAPPING_RULES_2, schema_version=None)
    self.assertIsNone(result)