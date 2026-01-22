import uuid
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_roles(self):
    role1 = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex}
    role2 = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex}
    token = fixture.V3Token()
    token.add_role(**role1)
    token.add_role(**role2)
    self.assertEqual(2, len(token['token']['roles']))
    self.assertIn(role1, token['token']['roles'])
    self.assertIn(role2, token['token']['roles'])