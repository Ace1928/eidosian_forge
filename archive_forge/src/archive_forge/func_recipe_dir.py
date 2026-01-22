import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
@recipe_dir.setter
def recipe_dir(self, recipe_dir: Path):
    self._recipe_dir = recipe_dir.resolve() if recipe_dir else None
    if self._recipe_dir:
        self.set_value('buildozer', 'recipe_dir', str(self._recipe_dir))