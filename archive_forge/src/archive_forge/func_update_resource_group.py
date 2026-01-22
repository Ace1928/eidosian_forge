import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
def update_resource_group(self, update_template, updated, created, deleted):
    stack_identifier = self.stack_create(template=self.template)
    group_resources = self.list_group_resources(stack_identifier, 'random_group', minimal=False)
    init_names = [res.physical_resource_id for res in group_resources]
    self.update_stack(stack_identifier, update_template)
    group_resources = self.list_group_resources(stack_identifier, 'random_group', minimal=False)
    updt_names = [res.physical_resource_id for res in group_resources]
    matched_names = set(updt_names) & set(init_names)
    self.assertEqual(updated, len(matched_names))
    self.assertEqual(created, len(set(updt_names) - set(init_names)))
    self.assertEqual(deleted, len(set(init_names) - set(updt_names)))