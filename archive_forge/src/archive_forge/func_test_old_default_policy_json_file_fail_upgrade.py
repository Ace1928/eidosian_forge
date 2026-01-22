import fixtures
import os.path
import tempfile
import yaml
from oslo_config import cfg
from oslo_config import fixture as config
from oslo_policy import opts as policy_opts
from oslo_serialization import jsonutils
from oslotest import base
from oslo_upgradecheck import common_checks
from oslo_upgradecheck import upgradecheck
def test_old_default_policy_json_file_fail_upgrade(self):
    self.conf.set_override('policy_file', 'policy.json', group='oslo_policy')
    tmpfilename = os.path.join(self.temp_dir.path, 'policy.json')
    with open(tmpfilename, 'w') as fh:
        jsonutils.dump(self.data, fh)
    self.assertEqual(upgradecheck.Code.FAILURE, self.cmd.check())