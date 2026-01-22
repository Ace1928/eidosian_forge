import copy
from unittest import mock
from openstackclient.identity.v3 import role_assignment
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_role_assignment_list_include_names(self):
    fake_role_assignment_a = copy.deepcopy(identity_fakes.ASSIGNMENT_WITH_PROJECT_ID_AND_USER_ID_INCLUDE_NAMES)
    fake_role_assignment_b = copy.deepcopy(identity_fakes.ASSIGNMENT_WITH_DOMAIN_ID_AND_USER_ID_INCLUDE_NAMES)
    self.role_assignments_mock.list.return_value = [fakes.FakeResource(None, fake_role_assignment_a, loaded=True), fakes.FakeResource(None, fake_role_assignment_b, loaded=True)]
    arglist = ['--names']
    verifylist = [('user', None), ('group', None), ('system', None), ('domain', None), ('project', None), ('role', None), ('effective', False), ('inherited', False), ('names', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.role_assignments_mock.list.assert_called_with(domain=None, system=None, group=None, effective=False, project=None, role=None, user=None, os_inherit_extension_inherited_to=None, include_names=True)
    collist = ('Role', 'User', 'Group', 'Project', 'Domain', 'System', 'Inherited')
    self.assertEqual(columns, collist)
    datalist1 = ((identity_fakes.role_name, '@'.join([identity_fakes.user_name, identity_fakes.domain_name]), '', '@'.join([identity_fakes.project_name, identity_fakes.domain_name]), '', '', False), (identity_fakes.role_name, '@'.join([identity_fakes.user_name, identity_fakes.domain_name]), '', '', identity_fakes.domain_name, '', False))
    self.assertEqual(tuple(data), datalist1)