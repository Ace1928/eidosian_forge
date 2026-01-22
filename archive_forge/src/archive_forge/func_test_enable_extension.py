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
def test_enable_extension(self):
    options = AppOptions(app_dir=self.tempdir())
    assert install_extension(self.mock_extension, app_options=options) is True
    assert disable_extension(self.pkg_names['extension'], app_options=options) is True
    assert enable_extension(self.pkg_names['extension'], app_options=options) is True
    info = get_app_info(app_options=options)
    assert '@jupyterlab/notebook-extension' not in info['disabled']
    name = self.pkg_names['extension']
    assert info['disabled'].get(name, False) is False
    assert check_extension(name, app_options=options)
    assert disable_extension('@jupyterlab/notebook-extension', app_options=options) is True
    assert check_extension(name, app_options=options)
    assert not check_extension('@jupyterlab/notebook-extension', app_options=options)