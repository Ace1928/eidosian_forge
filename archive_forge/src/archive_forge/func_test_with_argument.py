import os
import re
import subprocess
import sys
from pbr.tests import base
def test_with_argument(self):
    if os.name == 'nt':
        self.skipTest('Windows support is passthrough')
    stdout, _, return_code = self.run_setup('install', '--prefix=%s' % self.temp_dir)
    self._test_wsgi('pbr_test_wsgi', b'Foo Bar', ['--', '-c', 'Foo Bar'])