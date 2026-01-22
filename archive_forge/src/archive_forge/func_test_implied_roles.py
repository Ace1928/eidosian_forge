from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_implied_roles(self):
    initial_rule_count = len(self.client.inference_rules.list_inference_roles())
    self.create_roles()
    self.create_rules()
    rule_count = len(self.client.inference_rules.list_inference_roles())
    self.assertEqual(initial_rule_count + len(inference_rules), rule_count)