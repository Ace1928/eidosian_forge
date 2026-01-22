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
def load_sphinx_rtd_theme(self) -> None:
    """Load sphinx_rtd_theme theme (if installed)."""
    try:
        import sphinx_rtd_theme
        theme_path = sphinx_rtd_theme.get_html_theme_path()
        self.themes['sphinx_rtd_theme'] = path.join(theme_path, 'sphinx_rtd_theme')
    except ImportError:
        pass