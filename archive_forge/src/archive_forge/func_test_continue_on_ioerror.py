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
def test_continue_on_ioerror(self):
    """Assert we get all files processed.

        Assert we get all files processed even if there are processing
        exceptions.

        Test uses IOError as side effect of mock open.
        """
    with mock.patch('glob.glob') as m_ldir:
        m_ldir.return_value = ['/etc_etc/heat/environment.d/a.yaml', '/etc_etc/heat/environment.d/b.yaml']
        env_dir = '/etc_etc/heat/environment.d'
        env_content = '{}'
        env = environment.Environment({}, user_env=False)
        with mock.patch('heat.engine.environment.open', mock.mock_open(read_data=env_content), create=True) as m_open:
            m_open.side_effect = IOError
            environment.read_global_environment(env, env_dir)
    m_ldir.assert_called_once_with(env_dir + '/*')
    expected = [mock.call('%s/a.yaml' % env_dir), mock.call('%s/b.yaml' % env_dir)]
    self.assertEqual(expected, m_open.call_args_list)