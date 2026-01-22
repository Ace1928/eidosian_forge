import copy
from osc_lib.tests import utils as osc_lib_utils
from openstackclient import shell
from openstackclient.tests.unit.integ import base as test_base
from openstackclient.tests.unit import test_shell
def test_project_name_arg(self):
    _shell = shell.OpenStackShell()
    _shell.run('--os-project-name wsx extension list'.split())
    self.assertNotEqual(len(self.requests_mock.request_history), 0)
    self.assertEqual(test_base.V3_AUTH_URL, self.requests_mock.request_history[0].url)
    auth_req = self.requests_mock.request_history[1].json()
    self.assertEqual('wsx', auth_req['auth']['scope']['project']['name'])
    self.assertIsNone(auth_req['auth'].get('tenantId', None))
    self.assertIsNone(auth_req['auth'].get('tenantName', None))