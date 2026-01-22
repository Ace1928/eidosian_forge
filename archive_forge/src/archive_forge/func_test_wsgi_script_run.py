import os
import re
import subprocess
import sys
from pbr.tests import base
def test_wsgi_script_run(self):
    """Test that we install a runnable wsgi script.

        This test actually attempts to start and interact with the
        wsgi script in question to demonstrate that it's a working
        wsgi script using simple server.

        """
    if os.name == 'nt':
        self.skipTest('Windows support is passthrough')
    stdout, _, return_code = self.run_setup('install', '--prefix=%s' % self.temp_dir)
    self._check_wsgi_install_content(stdout)
    for cmd_name in self.cmd_names:
        self._test_wsgi(cmd_name, b'Hello World')