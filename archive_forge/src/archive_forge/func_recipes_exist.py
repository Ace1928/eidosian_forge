import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
def recipes_exist(self):
    if not self._recipe_dir:
        return False
    pyside_recipe_dir = Path(self.recipe_dir) / 'PySide6'
    shiboken_recipe_dir = Path(self.recipe_dir) / 'shiboken6'
    return pyside_recipe_dir.is_dir() and shiboken_recipe_dir.is_dir()