import configparser
import os
import shutil
import tempfile
from os import path
from typing import TYPE_CHECKING, Any, Dict, List
from zipfile import ZipFile
from sphinx import package_dir
from sphinx.errors import ThemeError
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.osutil import ensuredir
def is_archived_theme(filename: str) -> bool:
    """Check whether the specified file is an archived theme file or not."""
    try:
        with ZipFile(filename) as f:
            return THEMECONF in f.namelist()
    except Exception:
        return False