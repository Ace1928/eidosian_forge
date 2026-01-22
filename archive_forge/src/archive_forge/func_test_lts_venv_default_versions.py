import os.path
import pkg_resources
import shlex
import sys
import fixtures
import testtools
import textwrap
from pbr.tests import base
from pbr.tests import test_packaging
@testtools.skipUnless(os.environ.get('PBR_INTEGRATION', None) == '1', 'integration tests not enabled')
def test_lts_venv_default_versions(self):
    venv = self.useFixture(test_packaging.Venv('setuptools', modules=self.modules))
    bin_python = venv.python
    pbr = 'file://%s#egg=pbr' % PBR_ROOT
    self._run_cmd(bin_python, ['-m', 'pip', 'install', pbr], cwd=venv.path, allow_fail=False)