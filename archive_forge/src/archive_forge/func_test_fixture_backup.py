import re
from unittest import mock
from oslo_config import cfg
from oslo_db import options
from oslotest import base
from neutron_lib.api import attributes
from neutron_lib.api.definitions import port
from neutron_lib.callbacks import registry
from neutron_lib.db import model_base
from neutron_lib.db import resource_extend
from neutron_lib import fixture
from neutron_lib.placement import client as place_client
from neutron_lib.plugins import directory
from neutron_lib.tests.unit.api import test_attributes
def test_fixture_backup(self):
    fake_methods = {'a': 'A', 'b': 'B'}
    orig_methods = resource_extend._resource_extend_functions
    self.assertNotEqual(fake_methods, orig_methods)
    db_fixture = fixture.DBResourceExtendFixture(extended_functions=fake_methods)
    db_fixture.setUp()
    resource_extend.register_funcs('C', (lambda x: x,))
    self.assertNotEqual(orig_methods, resource_extend._resource_extend_functions)
    db_fixture.cleanUp()
    self.assertEqual(orig_methods, resource_extend._resource_extend_functions)