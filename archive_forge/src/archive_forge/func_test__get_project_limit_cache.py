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
def test__get_project_limit_cache(self, cache=True):
    project_id = uuid.uuid4().hex
    fix = self.useFixture(fixture.LimitFixture({'foo': 5}, {project_id: {'foo': 3}}))
    utils = limit._EnforcerUtils(cache=cache)
    foo_limit = utils._get_project_limit(project_id, 'foo')
    self.assertEqual(3, foo_limit.resource_limit)
    self.assertEqual(1, fix.mock_conn.limits.call_count)
    foo_limit = utils._get_project_limit(project_id, 'foo')
    count = 1 if cache else 2
    self.assertEqual(count, fix.mock_conn.limits.call_count)