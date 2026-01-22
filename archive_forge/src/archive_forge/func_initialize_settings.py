import atexit
import json
import os
import shutil
import sys
import tempfile
from os import path as osp
from os.path import join as pjoin
from stat import S_IRGRP, S_IROTH, S_IRUSR
from tempfile import TemporaryDirectory
from unittest.mock import patch
import jupyter_core
import jupyterlab_server
from ipykernel.kernelspec import write_kernel_spec
from jupyter_server.serverapp import ServerApp
from jupyterlab_server.process_app import ProcessApp
from traitlets import default
def initialize_settings(self):
    self.env_patch = TestEnv()
    self.env_patch.start()
    ProcessApp.__init__(self)
    self.settings['allow_origin'] = ProcessTestApp.allow_origin
    self.static_dir = self.static_paths[0]
    self.template_dir = self.template_paths[0]
    self.schemas_dir = _create_schemas_dir()
    self.user_settings_dir = _create_user_settings_dir()
    self.workspaces_dir = _create_workspaces_dir()
    self._install_default_kernels()
    self.settings['kernel_manager'].default_kernel_name = 'echo'
    super().initialize_settings()