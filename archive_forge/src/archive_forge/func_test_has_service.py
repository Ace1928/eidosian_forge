import os
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack.tests.functional import base
def test_has_service(self):
    if os.environ.get('OPENSTACKSDK_HAS_{env}'.format(env=self.env), '0') == '1':
        self.assertTrue(self.user_cloud.has_service(self.service))