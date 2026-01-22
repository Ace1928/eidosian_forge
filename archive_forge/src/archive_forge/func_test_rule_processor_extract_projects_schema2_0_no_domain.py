import copy
import uuid
from keystone.exception import ValidationError
from keystone.federation import utils
from keystone.tests import unit
def test_rule_processor_extract_projects_schema2_0_no_domain(self):
    projects_list = [{'name': 'project1'}]
    identity_values = {'projects': projects_list}
    result = self.rule_processor_schema_2_0.extract_projects(identity_values)
    self.assertEqual(projects_list, result)