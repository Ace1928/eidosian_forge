from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def create_rules(self):
    roles = self.role_dict()
    for prior, implied in inference_rules.items():
        rule = fixtures.InferenceRule(self.client, roles[prior], roles[implied])
        self.useFixture(rule)