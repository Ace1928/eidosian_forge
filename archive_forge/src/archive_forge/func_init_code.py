import glob
from itertools import chain
import os
import sys
from traitlets.config.application import boolean_flag
from traitlets.config.configurable import Configurable
from traitlets.config.loader import Config
from IPython.core.application import SYSTEM_CONFIG_DIRS, ENV_CONFIG_DIRS
from IPython.core import pylabtools
from IPython.utils.contexts import preserve_keys
from IPython.utils.path import filefind
from traitlets import (
from IPython.terminal import pt_inputhooks
def init_code(self):
    """run the pre-flight code, specified via exec_lines"""
    self._run_startup_files()
    self._run_exec_lines()
    self._run_exec_files()
    if self.hide_initial_ns:
        self.shell.user_ns_hidden.update(self.shell.user_ns)
    self._run_cmd_line_code()
    self._run_module()
    sys.stdout.flush()
    sys.stderr.flush()
    self.shell._sys_modules_keys = set(sys.modules.keys())