import logging
import subprocess
from os.path import join as pjoin
from unittest.mock import patch
from jupyterlab import commands
from .test_jupyterlab import AppHandlerTest
def test_populate_staging(self):
    with patch('subprocess.check_output') as check_output:
        yarn_registry = 'https://private.yarn/manager'
        check_output.return_value = b'\n'.join([b'{"type":"info","data":"yarn config"}', b'{"type":"inspect","data":{"registry":"' + bytes(yarn_registry, 'utf-8') + b'"}}', b'{"type":"info","data":"npm config"}', b'{"type":"inspect","data":{"registry":"' + bytes(yarn_registry, 'utf-8') + b'"}}'])
        staging = pjoin(self.app_dir, 'staging')
        handler = commands._AppHandler(commands.AppOptions())
        handler._populate_staging()
        lock_path = pjoin(staging, 'yarn.lock')
        with open(lock_path) as f:
            lock = f.read()
        self.assertNotIn(commands.YARN_DEFAULT_REGISTRY, lock)
        self.assertNotIn(yarn_registry, lock)