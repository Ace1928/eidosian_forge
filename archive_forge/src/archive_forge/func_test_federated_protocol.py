from keystone.common import sql
from keystone.tests.unit import test_backend_sql
def test_federated_protocol(self):
    cols = (('id', sql.String, 64), ('idp_id', sql.String, 64), ('mapping_id', sql.String, 64), ('remote_id_attribute', sql.String, 64))
    self.assertExpectedSchema('federation_protocol', cols)