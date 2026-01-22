import os.path
from unittest import mock
import fixtures
from oslo_config import cfg
from heat.common import environment_format
from heat.common import exception
from heat.engine import environment
from heat.engine import resources
from heat.engine.resources.aws.ec2 import instance
from heat.engine.resources.openstack.nova import server
from heat.engine import support
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
def test_env_resources_override_plugins(self):
    g_env_content = '\n        resource_registry:\n          "OS::Nova::Server": "file:///not_really_here.yaml"\n        '
    envdir = self.useFixture(fixtures.TempDir())
    envfile = os.path.join(envdir.path, 'test.yaml')
    with open(envfile, 'w+') as ef:
        ef.write(g_env_content)
    cfg.CONF.set_override('environment_dir', envdir.path)
    g_env = environment.Environment({}, user_env=False)
    resources._load_global_environment(g_env)
    self.assertEqual('file:///not_really_here.yaml', g_env.get_resource_info('OS::Nova::Server').value)