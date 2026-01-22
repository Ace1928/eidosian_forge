import os
import unittest
from distutils.command.install_scripts import install_scripts
from distutils.core import Distribution
from distutils.tests import support
def test_installation(self):
    source = self.mkdtemp()
    expected = []

    def write_script(name, text):
        expected.append(name)
        f = open(os.path.join(source, name), 'w')
        try:
            f.write(text)
        finally:
            f.close()
    write_script('script1.py', '#! /usr/bin/env python2.3\n# bogus script w/ Python sh-bang\npass\n')
    write_script('script2.py', '#!/usr/bin/python\n# bogus script w/ Python sh-bang\npass\n')
    write_script('shell.sh', '#!/bin/sh\n# bogus shell script w/ sh-bang\nexit 0\n')
    target = self.mkdtemp()
    dist = Distribution()
    dist.command_obj['build'] = support.DummyCommand(build_scripts=source)
    dist.command_obj['install'] = support.DummyCommand(install_scripts=target, force=1, skip_build=1)
    cmd = install_scripts(dist)
    cmd.finalize_options()
    cmd.run()
    installed = os.listdir(target)
    for name in expected:
        self.assertIn(name, installed)