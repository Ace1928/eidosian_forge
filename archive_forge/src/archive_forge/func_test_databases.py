from openstack.database.v1 import _proxy
from openstack.database.v1 import database
from openstack.database.v1 import flavor
from openstack.database.v1 import instance
from openstack.database.v1 import user
from openstack.tests.unit import test_proxy_base
def test_databases(self):
    self.verify_list(self.proxy.databases, database.Database, method_args=['id'], expected_args=[], expected_kwargs={'instance_id': 'id'})