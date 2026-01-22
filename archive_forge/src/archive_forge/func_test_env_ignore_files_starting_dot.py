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
def test_env_ignore_files_starting_dot(self):
    g_env_content = ''
    envdir = self.useFixture(fixtures.TempDir())
    with open(os.path.join(envdir.path, 'a.yaml'), 'w+') as ef:
        ef.write(g_env_content)
    with open(os.path.join(envdir.path, '.test.yaml'), 'w+') as ef:
        ef.write(g_env_content)
    with open(os.path.join(envdir.path, 'b.yaml'), 'w+') as ef:
        ef.write(g_env_content)
    cfg.CONF.set_override('environment_dir', envdir.path)
    g_env = environment.Environment({}, user_env=False)
    with mock.patch('heat.engine.environment.open', mock.mock_open(read_data=g_env_content), create=True) as m_open:
        resources._load_global_environment(g_env)
    expected = [mock.call('%s/a.yaml' % envdir.path), mock.call('%s/b.yaml' % envdir.path)]
    call_list = m_open.call_args_list
    expected.sort()
    call_list.sort()
    self.assertEqual(expected, call_list)