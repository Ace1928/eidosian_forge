from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import project
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_project_show_parents_and_children(self):
    self.project = identity_fakes.FakeProject.create_one_project(attrs={'parent_id': self.project.parent_id, 'parents': [{'project': {'id': self.project.parent_id}}], 'subtree': [{'project': {'id': 'children-id'}}]})
    self.projects_mock.get.return_value = self.project
    arglist = [self.project.id, '--parents', '--children']
    verifylist = [('project', self.project.id), ('parents', True), ('children', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.app.client_manager.identity.tokens.get_token_data.return_value = {'token': {'project': {'domain': {}, 'name': parsed_args.project, 'id': parsed_args.project}}}
    columns, data = self.cmd.take_action(parsed_args)
    self.projects_mock.get.assert_has_calls([call(self.project.id), call(self.project.id, parents_as_ids=True, subtree_as_ids=True)])
    collist = ('description', 'domain_id', 'enabled', 'id', 'is_domain', 'name', 'parent_id', 'parents', 'subtree', 'tags')
    self.assertEqual(columns, collist)
    datalist = (self.project.description, self.project.domain_id, self.project.enabled, self.project.id, self.project.is_domain, self.project.name, self.project.parent_id, [{'project': {'id': self.project.parent_id}}], [{'project': {'id': 'children-id'}}], self.project.tags)
    self.assertEqual(data, datalist)