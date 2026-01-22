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
def test_app_dir_disable_sys_prefix(self):
    app_dir = self.tempdir()
    options = AppOptions(app_dir=app_dir, use_sys_dir=False)
    if os.path.exists(self.app_dir):
        os.removedirs(self.app_dir)
    assert install_extension(self.mock_extension) is True
    path = pjoin(app_dir, 'extensions', '*.tgz')
    assert not glob.glob(path)
    extensions = get_app_info(app_options=options)['extensions']
    ext_name = self.pkg_names['extension']
    assert ext_name not in extensions
    assert not check_extension(ext_name, app_options=options)