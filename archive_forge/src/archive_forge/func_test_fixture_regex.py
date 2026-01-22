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
@mock.patch.object(fixture.warnings, 'filterwarnings')
def test_fixture_regex(self, mock_filterwarnings):
    module_re = ['^neutron\\.']
    warn_fixture = fixture.WarningsFixture(module_re=module_re)
    warn_fixture.setUp()
    call_re = mock_filterwarnings.mock_calls[0][2]['module']
    self.assertEqual('^neutron_lib\\.|^neutron\\.', call_re)
    self.assertIsNotNone(re.compile(call_re))
    self.assertIsNotNone(re.search(call_re, 'neutron.db.blah'))
    self.assertIsNotNone(re.search(call_re, 'neutron_lib.db.blah'))
    self.assertIsNone(re.search(call_re, 'neutron_dynamic_routing.db.blah'))