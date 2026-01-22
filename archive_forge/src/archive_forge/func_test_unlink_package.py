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
def test_unlink_package(self):
    target = self.mock_package
    assert link_package(target) is True
    assert unlink_package(target) is True
    linked = get_app_info()['linked_packages']
    name = self.pkg_names['package']
    assert name not in linked
    assert not check_extension(name)