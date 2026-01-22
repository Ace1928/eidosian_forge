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
def test_install_compatible(self):
    core_data = _get_default_core_data()
    current_app_dep = core_data['dependencies']['@jupyterlab/application']

    def _gen_dep(ver):
        return {'dependencies': {'@jupyterlab/application': ver}}

    def _mock_metadata(registry, name, logger):
        assert name == 'mockextension'
        return {'name': name, 'versions': {'0.9.0': _gen_dep(current_app_dep), '1.0.0': _gen_dep(current_app_dep), '1.1.0': _gen_dep(current_app_dep), '2.0.0': _gen_dep('^2000.0.0'), '2.0.0-b0': _gen_dep(current_app_dep), '2.1.0-b0': _gen_dep('^2000.0.0'), '2.1.0': _gen_dep('^2000.0.0')}}

    def _mock_extract(self, source, tempdir, *args, **kwargs):
        data = {'name': source, 'version': '2.1.0', 'jupyterlab': {'extension': True}, 'jupyterlab_extracted_files': ['index.js']}
        data.update(_gen_dep('^2000.0.0'))
        info = {'source': source, 'is_dir': False, 'data': data, 'name': source, 'version': data['version'], 'filename': 'mockextension.tgz', 'path': pjoin(tempdir, 'mockextension.tgz')}
        return info

    class Success(Exception):
        pass

    def _mock_install(self, name, *args, **kwargs):
        assert name in ('mockextension', 'mockextension@1.1.0')
        if name == 'mockextension@1.1.0':
            raise Success()
        return orig_install(self, name, *args, **kwargs)
    p1 = patch.object(commands, '_fetch_package_metadata', _mock_metadata)
    p2 = patch.object(commands._AppHandler, '_extract_package', _mock_extract)
    p3 = patch.object(commands._AppHandler, '_install_extension', _mock_install)
    with p1, p2:
        orig_install = commands._AppHandler._install_extension
        with p3, pytest.raises(Success):
            assert install_extension('mockextension') is True