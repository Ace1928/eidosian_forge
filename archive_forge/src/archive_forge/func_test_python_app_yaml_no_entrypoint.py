import os
import textwrap
import unittest
from gae_ext_runtime import comm
from gae_ext_runtime import ext_runtime
from gae_ext_runtime import testutil
def test_python_app_yaml_no_entrypoint(self):
    self.write_file('test.py', 'test file')
    config = testutil.AppInfoFake(runtime='python')
    self.generate_configs(appinfo=config, deploy=True)
    self.assert_file_exists_with_contents('Dockerfile', self.DOCKERFILE_PREAMBLE + self.DOCKERFILE_VIRTUALENV_TEMPLATE.format(python_version='') + self.DOCKERFILE_INSTALL_APP + 'CMD my_entrypoint\n')
    self.assertEqual(set(os.listdir(self.temp_path)), {'test.py', 'Dockerfile', '.dockerignore'})