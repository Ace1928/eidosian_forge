import os
import sys
import unittest
import site
from test.support import captured_stdout, requires_subprocess
from distutils import sysconfig
from distutils.command.install import install, HAS_USER_SITE
from distutils.command import install as install_module
from distutils.command.build_ext import build_ext
from distutils.command.install import INSTALL_SCHEMES
from distutils.core import Distribution
from distutils.errors import DistutilsOptionError
from distutils.extension import Extension
from distutils.tests import support
from test import support as test_support
def test_home_installation_scheme(self):
    builddir = self.mkdtemp()
    destination = os.path.join(builddir, 'installation')
    dist = Distribution({'name': 'foopkg'})
    dist.script_name = os.path.join(builddir, 'setup.py')
    dist.command_obj['build'] = support.DummyCommand(build_base=builddir, build_lib=os.path.join(builddir, 'lib'))
    cmd = install(dist)
    cmd.home = destination
    cmd.ensure_finalized()
    self.assertEqual(cmd.install_base, destination)
    self.assertEqual(cmd.install_platbase, destination)

    def check_path(got, expected):
        got = os.path.normpath(got)
        expected = os.path.normpath(expected)
        self.assertEqual(got, expected)
    libdir = os.path.join(destination, 'lib', 'python')
    check_path(cmd.install_lib, libdir)
    platlibdir = os.path.join(destination, sys.platlibdir, 'python')
    check_path(cmd.install_platlib, platlibdir)
    check_path(cmd.install_purelib, libdir)
    check_path(cmd.install_headers, os.path.join(destination, 'include', 'python', 'foopkg'))
    check_path(cmd.install_scripts, os.path.join(destination, 'bin'))
    check_path(cmd.install_data, destination)