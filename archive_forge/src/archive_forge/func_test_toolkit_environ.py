import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
def test_toolkit_environ(self):
    test_args = ['something']
    test_environ = {'ETS_TOOLKIT': 'test'}
    with mock_sys_argv(test_args):
        with mock_os_environ(test_environ):
            toolkit = self.ETSConfig.toolkit
    self.assertEqual(toolkit, 'test')