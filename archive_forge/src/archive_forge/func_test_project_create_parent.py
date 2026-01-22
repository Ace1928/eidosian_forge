from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import project
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_project_create_parent(self):
    self.parent = identity_fakes.FakeProject.create_one_project()
    self.project = identity_fakes.FakeProject.create_one_project(attrs={'domain_id': self.domain.id, 'parent_id': self.parent.id})
    self.projects_mock.get.return_value = self.parent
    self.projects_mock.create.return_value = self.project
    arglist = ['--domain', self.project.domain_id, '--parent', self.parent.name, self.project.name]
    verifylist = [('domain', self.project.domain_id), ('parent', self.parent.name), ('enable', False), ('disable', False), ('name', self.project.name), ('tags', [])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'name': self.project.name, 'domain': self.project.domain_id, 'parent': self.parent.id, 'description': None, 'enabled': True, 'tags': [], 'options': {}}
    self.projects_mock.create.assert_called_with(**kwargs)
    collist = ('description', 'domain_id', 'enabled', 'id', 'is_domain', 'name', 'parent_id', 'tags')
    self.assertEqual(columns, collist)
    datalist = (self.project.description, self.project.domain_id, self.project.enabled, self.project.id, self.project.is_domain, self.project.name, self.parent.id, self.project.tags)
    self.assertEqual(data, datalist)