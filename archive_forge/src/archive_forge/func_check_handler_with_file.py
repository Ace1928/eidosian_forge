import base64
import os
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import pytest
from jupyter_console.ptshell import ZMQTerminalInteractiveShell
def check_handler_with_file(self, inpath, handler):
    shell = self.shell
    configname = '{0}_image_handler'.format(handler)
    funcname = 'handle_image_{0}'.format(handler)
    assert hasattr(shell, configname)
    assert hasattr(shell, funcname)
    with TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, 'data')
        cmd = [sys.executable, SCRIPT_PATH, inpath, outpath]
        setattr(shell, configname, cmd)
        getattr(shell, funcname)(self.data, self.mime)
        with open(outpath, 'rb') as file:
            transferred = file.read()
    self.assertEqual(transferred, self.raw)