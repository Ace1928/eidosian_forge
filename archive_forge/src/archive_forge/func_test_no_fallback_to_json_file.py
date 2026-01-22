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
def test_no_fallback_to_json_file(self):
    tmpfilename = 'policy.yaml'
    self.conf.set_override('policy_file', tmpfilename, group='oslo_policy')
    jsonfile = os.path.join(self.tmpdir.path, 'policy.json')
    with open(jsonfile, 'w') as fh:
        jsonutils.dump(self.data, fh)
    selected_policy_file = policy.pick_default_policy_file(self.conf, fallback_to_json_file=False)
    self.assertEqual(self.conf.oslo_policy.policy_file, tmpfilename)
    self.assertEqual(selected_policy_file, tmpfilename)