import os
import unittest
from distutils.command.build_scripts import build_scripts
from distutils.core import Distribution
from distutils import sysconfig
from distutils.tests import support
def write_sample_scripts(self, dir):
    expected = []
    expected.append('script1.py')
    self.write_script(dir, 'script1.py', '#! /usr/bin/env python2.3\n# bogus script w/ Python sh-bang\npass\n')
    expected.append('script2.py')
    self.write_script(dir, 'script2.py', '#!/usr/bin/python\n# bogus script w/ Python sh-bang\npass\n')
    expected.append('shell.sh')
    self.write_script(dir, 'shell.sh', '#!/bin/sh\n# bogus shell script w/ sh-bang\nexit 0\n')
    return expected