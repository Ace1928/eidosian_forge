import glob
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from os.path import join as pjoin
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import pytest
from jupyter_core import paths
from jupyterlab import commands
from jupyterlab.commands import (
from jupyterlab.coreconfig import CoreConfig, _get_default_core_data
def test_update_multiple(self):
    installed = []

    def _mock_install(self, name, *args, **kwargs):
        installed.append(name[0] + name[1:].split('@')[0])
        return {'name': name, 'is_dir': False, 'path': 'foo/bar/' + name}

    def _mock_latest(self, name):
        return '10000.0.0'
    p1 = patch.object(commands._AppHandler, '_install_extension', _mock_install)
    p2 = patch.object(commands._AppHandler, '_latest_compatible_package_version', _mock_latest)
    install_extension(self.mock_extension)
    install_extension(self.mock_mimeextension)
    with p1, p2:
        assert update_extension(self.pkg_names['extension']) is True
        assert update_extension(self.pkg_names['mimeextension']) is True
    assert installed == [self.pkg_names['extension'], self.pkg_names['mimeextension']]