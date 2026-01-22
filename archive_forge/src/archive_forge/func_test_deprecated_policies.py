import os.path
import ddt
from oslo_config import fixture as config_fixture
from oslo_policy import policy as base_policy
from heat.common import exception
from heat.common import policy
from heat.tests import common
from heat.tests import utils
@ddt.file_data('policy/test_deprecated_access.yaml')
@ddt.unpack
def test_deprecated_policies(self, **kwargs):
    self.fixture.config(group='oslo_policy', enforce_scope=False)
    self.fixture.config(group='oslo_policy', enforce_new_defaults=False)
    self._test_legacy_rbac_policies(**kwargs)