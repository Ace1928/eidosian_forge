import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
@local_libs.setter
def local_libs(self, local_libs):
    self._local_libs = local_libs
    self.set_value('buildozer', 'local_libs', ','.join(local_libs))