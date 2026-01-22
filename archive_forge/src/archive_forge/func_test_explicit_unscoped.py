from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_explicit_unscoped(self):
    post_data = {'identity': {'methods': ['password'], 'password': {'user': {'name': 'admin', 'domain': {'name': 'Default'}, 'password': 'devstacker'}}}, 'scope': 'unscoped'}
    schema.validate_issue_token_auth(post_data)