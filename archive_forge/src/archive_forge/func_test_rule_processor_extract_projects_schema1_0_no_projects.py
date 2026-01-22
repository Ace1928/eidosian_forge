import copy
import uuid
from keystone.exception import ValidationError
from keystone.federation import utils
from keystone.tests import unit
def test_rule_processor_extract_projects_schema1_0_no_projects(self):
    result = self.rule_processor.extract_projects({})
    self.assertEqual([], result)