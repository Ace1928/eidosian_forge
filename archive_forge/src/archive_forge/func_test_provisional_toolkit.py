import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
def test_provisional_toolkit(self):
    test_args = []
    test_environ = {}
    with mock_sys_argv(test_args):
        with mock_os_environ(test_environ):
            repr(self.ETSConfig.toolkit)
            with self.ETSConfig.provisional_toolkit('test_direct'):
                toolkit = self.ETSConfig.toolkit
                self.assertEqual(toolkit, 'test_direct')
    toolkit = self.ETSConfig.toolkit
    self.assertEqual(toolkit, 'test_direct')