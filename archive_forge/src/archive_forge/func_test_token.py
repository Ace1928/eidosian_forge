from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_token(self):
    p = {'identity': {'methods': ['token'], 'token': {'id': 'something'}}}
    schema.validate_issue_token_auth(p)