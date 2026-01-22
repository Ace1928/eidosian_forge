import os
import unittest
from distutils.command.build_scripts import build_scripts
from distutils.core import Distribution
from distutils import sysconfig
from distutils.tests import support
def test_version_int(self):
    source = self.mkdtemp()
    target = self.mkdtemp()
    expected = self.write_sample_scripts(source)
    cmd = self.get_build_scripts_cmd(target, [os.path.join(source, fn) for fn in expected])
    cmd.finalize_options()
    old = sysconfig.get_config_vars().get('VERSION')
    sysconfig._config_vars['VERSION'] = 4
    try:
        cmd.run()
    finally:
        if old is not None:
            sysconfig._config_vars['VERSION'] = old
    built = os.listdir(target)
    for name in expected:
        self.assertIn(name, built)