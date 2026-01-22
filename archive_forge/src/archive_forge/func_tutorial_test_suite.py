import doctest
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import ClassVar, List
from unittest import SkipTest, expectedFailure, skipIf
from unittest import TestCase as _TestCase
def tutorial_test_suite():
    tutorial = ['introduction', 'file-format', 'repo', 'object-store', 'remote', 'conclusion']
    tutorial_files = [f'../../docs/tutorial/{name}.txt' for name in tutorial]
    to_restore = []

    def overrideEnv(name, value):
        oldval = os.environ.get(name)
        if value is not None:
            os.environ[name] = value
        else:
            del os.environ[name]
        to_restore.append((name, oldval))

    def setup(test):
        test.__old_cwd = os.getcwd()
        test.tempdir = tempfile.mkdtemp()
        test.globs.update({'tempdir': test.tempdir})
        os.chdir(test.tempdir)
        overrideEnv('HOME', '/nonexistent')
        overrideEnv('GIT_CONFIG_NOSYSTEM', '1')

    def teardown(test):
        os.chdir(test.__old_cwd)
        shutil.rmtree(test.tempdir)
        for name, oldval in to_restore:
            if oldval is not None:
                os.environ[name] = oldval
            else:
                del os.environ[name]
        to_restore.clear()
    return doctest.DocFileSuite(*tutorial_files, module_relative=True, package='dulwich.tests', setUp=setup, tearDown=teardown)