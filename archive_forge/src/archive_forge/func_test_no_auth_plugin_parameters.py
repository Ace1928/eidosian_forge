from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_no_auth_plugin_parameters(self):
    post_data = {'identity': {'methods': ['password']}}
    schema.validate_issue_token_auth(post_data)