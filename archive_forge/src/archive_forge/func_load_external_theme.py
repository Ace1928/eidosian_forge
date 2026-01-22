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
def load_external_theme(self, name: str) -> None:
    """Try to load a theme using entry_points.

        Sphinx refers to ``sphinx_themes`` entry_points.
        """
    theme_entry_points = entry_points(group='sphinx.html_themes')
    try:
        entry_point = theme_entry_points[name]
        self.app.registry.load_extension(self.app, entry_point.module)
        self.app.config.post_init_values()
        return
    except KeyError:
        pass