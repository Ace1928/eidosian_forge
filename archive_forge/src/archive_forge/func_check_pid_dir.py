import os
import shutil
import errno
from pathlib import Path
from traitlets.config.configurable import LoggingConfigurable
from ..paths import get_ipython_package_dir
from ..utils.path import expand_path, ensure_dir_exists
from traitlets import Unicode, Bool, observe
@observe('pid_dir')
def check_pid_dir(self, change=None):
    self._mkdir(self.pid_dir, 16832)