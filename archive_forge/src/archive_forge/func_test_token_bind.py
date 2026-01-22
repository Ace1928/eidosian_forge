import uuid
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_token_bind(self):
    name1 = uuid.uuid4().hex
    data1 = uuid.uuid4().hex
    name2 = uuid.uuid4().hex
    data2 = {uuid.uuid4().hex: uuid.uuid4().hex}
    token = fixture.V3Token()
    token.set_bind(name1, data1)
    token.set_bind(name2, data2)
    self.assertEqual({name1: data1, name2: data2}, token['token']['bind'])