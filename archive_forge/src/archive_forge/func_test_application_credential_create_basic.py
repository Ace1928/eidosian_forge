import copy
import json
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import application_credential
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_application_credential_create_basic(self):
    self.app_creds_mock.create.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.APP_CRED_BASIC), loaded=True)
    name = identity_fakes.app_cred_name
    arglist = [name]
    verifylist = [('name', identity_fakes.app_cred_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'secret': None, 'roles': [], 'expires_at': None, 'description': None, 'unrestricted': False, 'access_rules': None}
    self.app_creds_mock.create.assert_called_with(name, **kwargs)
    collist = ('access_rules', 'description', 'expires_at', 'id', 'name', 'project_id', 'roles', 'secret', 'unrestricted')
    self.assertEqual(collist, columns)
    datalist = (None, None, None, identity_fakes.app_cred_id, identity_fakes.app_cred_name, identity_fakes.project_id, identity_fakes.role_name, identity_fakes.app_cred_secret, False)
    self.assertEqual(datalist, data)