from heat_integrationtests.functional import functional_base
def test_nested_resources_replace(self):
    files = {'nested1.yaml': self.template_nested1, 'nested2.yaml': self.template_nested2}
    self.stack_identifier = self.stack_create(template=self.template_nested_parent, files=files)
    parent_none = self.template_nested_parent.replace('nested1.yaml', 'OS::Heat::None')
    result = self.preview_update_stack(self.stack_identifier, template=parent_none, show_nested=True)
    changes = result['resource_changes']
    self.assertEqual(1, len(changes['replaced']))
    self.assertEqual('nested1', changes['replaced'][0]['resource_name'])
    self.assertEqual(2, len(changes['deleted']))
    d_random = self._get_by_resource_name(changes, 'random', 'deleted')
    self.assertEqual('nested2', d_random['parent_resource'])
    d_nested2 = self._get_by_resource_name(changes, 'nested2', 'deleted')
    self.assertEqual('nested1', d_nested2['parent_resource'])
    self.assert_empty_sections(changes, ['updated', 'unchanged', 'added'])