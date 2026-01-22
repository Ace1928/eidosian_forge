import sqlalchemy as sa
from neutron_lib import context
from neutron_lib.db import model_base
from neutron_lib.tests.unit.db import _base as db_base
def test_model_base(self):
    foo = TestTable(name='meh')
    self.assertEqual('meh', foo.name)
    self.assertIn('meh', str(foo))
    cols = [k for k, _v in foo]
    self.assertIn('name', cols)