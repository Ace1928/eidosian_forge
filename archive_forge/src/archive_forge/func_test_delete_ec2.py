from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_delete_ec2(self):
    user = fixtures.User(self.client, self.project_domain_id)
    self.useFixture(user)
    project = fixtures.Project(self.client, self.project_domain_id)
    self.useFixture(project)
    ec2 = self.client.ec2.create(user.id, project.id)
    self.client.ec2.delete(user.id, ec2.access)
    self.assertRaises(http.NotFound, self.client.ec2.get, user.id, ec2.access)