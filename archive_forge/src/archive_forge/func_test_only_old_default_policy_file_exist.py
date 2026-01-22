import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
def test_only_old_default_policy_file_exist(self):
    self.conf.set_override('policy_file', 'policy.yaml', group='oslo_policy')
    tmpfilename = os.path.join(self.tmpdir.path, 'policy.json')
    with open(tmpfilename, 'w') as fh:
        jsonutils.dump(self.data, fh)
    selected_policy_file = policy.pick_default_policy_file(self.conf)
    self.assertEqual(self.conf.oslo_policy.policy_file, 'policy.yaml')
    self.assertEqual(selected_policy_file, 'policy.json')