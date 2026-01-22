import os
import re
import subprocess
import sys
from pbr.tests import base
def test_wsgi_script_install(self):
    """Test that we install a non-pkg-resources wsgi script."""
    if os.name == 'nt':
        self.skipTest('Windows support is passthrough')
    stdout, _, return_code = self.run_setup('install', '--prefix=%s' % self.temp_dir)
    self._check_wsgi_install_content(stdout)