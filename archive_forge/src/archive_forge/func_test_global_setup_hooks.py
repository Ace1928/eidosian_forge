import os
from testtools import matchers
from testtools import skipUnless
from pbr import testr_command
from pbr.tests import base
from pbr.tests import util
def test_global_setup_hooks(self):
    """Test setup_hooks.

        Test that setup_hooks listed in the [global] section of setup.cfg are
        executed in order.
        """
    stdout, _, return_code = self.run_setup('egg_info')
    assert 'test_hook_1\ntest_hook_2' in stdout
    assert return_code == 0