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
@pytest.mark.slow
def test_build_custom(self):
    assert install_extension(self.mock_extension) is True
    build(name='foo', version='1.0', static_url='bar')
    entry = pjoin(self.app_dir, 'static', 'index.out.js')
    with open(entry) as fid:
        data = fid.read()
    assert self.pkg_names['extension'] in data
    pkg = pjoin(self.app_dir, 'static', 'package.json')
    with open(pkg) as fid:
        data = json.load(fid)
    assert data['jupyterlab']['name'] == 'foo'
    assert data['jupyterlab']['version'] == '1.0'
    assert data['jupyterlab']['staticUrl'] == 'bar'