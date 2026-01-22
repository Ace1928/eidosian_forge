from unittest import mock
import uuid
from openstack.identity.v3 import endpoint
from openstack.identity.v3 import limit as klimit
from openstack.identity.v3 import registered_limit
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslotest import base
from oslo_limit import exception
from oslo_limit import fixture
from oslo_limit import limit
from oslo_limit import opts
def test_project_id_must_be_a_string(self):
    enforcer = limit.Enforcer(self._get_usage_for_project)
    invalid_delta_types = [{}, 5, 5.1, True, False, [], None, '']
    for invalid_project_id in invalid_delta_types:
        self.assertRaises(ValueError, enforcer.enforce, invalid_project_id, {})