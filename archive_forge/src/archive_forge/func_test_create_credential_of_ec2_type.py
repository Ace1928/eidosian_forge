import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_create_credential_of_ec2_type(self):
    user = fixtures.User(self.client, self.test_domain.id)
    self.useFixture(user)
    credential_ref = {'user': user.id, 'type': 'ec2', 'blob': '{"access":"' + uuid.uuid4().hex + '","secret":"secretKey"}'}
    self.assertRaises(http.BadRequest, self.client.credentials.create, **credential_ref)
    project = fixtures.Project(self.client, self.test_domain.id)
    self.useFixture(project)
    credential_ref = {'user': user.id, 'type': 'ec2', 'blob': '{"access":"' + uuid.uuid4().hex + '","secret":"secretKey"}', 'project': project.id}
    credential = self.client.credentials.create(**credential_ref)
    self.addCleanup(self.client.credentials.delete, credential)
    self.check_credential(credential, credential_ref)