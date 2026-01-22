from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def role_dict(self):
    roles = {role.name: role.id for role in self.client.roles.list()}
    return roles