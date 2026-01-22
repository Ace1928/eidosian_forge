import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
def test_provisional_toolkit_exception(self):
    test_args = []
    test_environ = {'ETS_TOOLKIT': ''}
    with mock_sys_argv(test_args):
        with mock_os_environ(test_environ):
            try:
                with self.ETSConfig.provisional_toolkit('test_direct'):
                    toolkit = self.ETSConfig.toolkit
                    self.assertEqual(toolkit, 'test_direct')
                    raise ETSToolkitError('Test exception')
            except ETSToolkitError as exc:
                if not exc.message == 'Test exception':
                    raise
            toolkit = self.ETSConfig.toolkit
            self.assertEqual(toolkit, '')