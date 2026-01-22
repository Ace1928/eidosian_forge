import os
import shutil
import errno
from pathlib import Path
from traitlets.config.configurable import LoggingConfigurable
from ..paths import get_ipython_package_dir
from ..utils.path import expand_path, ensure_dir_exists
from traitlets import Unicode, Bool, observe
@observe('startup_dir')
def check_startup_dir(self, change=None):
    self._mkdir(self.startup_dir)
    readme = os.path.join(self.startup_dir, 'README')
    src = os.path.join(get_ipython_package_dir(), u'core', u'profile', u'README_STARTUP')
    if not os.path.exists(src):
        self.log.warning('Could not copy README_STARTUP to startup dir. Source file %s does not exist.', src)
    if os.path.exists(src) and (not os.path.exists(readme)):
        shutil.copy(src, readme)