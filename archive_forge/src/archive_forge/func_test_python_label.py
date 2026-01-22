import os
import textwrap
import unittest
from gae_ext_runtime import comm
from gae_ext_runtime import ext_runtime
from gae_ext_runtime import testutil
def test_python_label(self):
    """Test that a LABEL directive is added to the Dockerfile"""
    self.write_file('test.py', 'test file')
    config = testutil.AppInfoFake(runtime='python', entrypoint='run_me_some_python!', runtime_config=dict(python_version='3'))
    cfg_files = self.generate_config_data(appinfo=config, deploy=True)
    dockerfile = [f for f in cfg_files if f.filename == 'Dockerfile'][0]
    self.assertIn('LABEL python_version=python3.6\n', dockerfile.contents)