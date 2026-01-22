import plistlib
import shlex
import subprocess
from os import environ
from os import path
from subprocess import CalledProcessError, PIPE, STDOUT
from typing import Any
import sphinx
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.errors import SphinxError
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.fileutil import copy_asset, copy_asset_file
from sphinx.util.matching import Matcher
from sphinx.util.osutil import ensuredir, make_filename
def copy_applehelp_icon(self, resources_dir: str) -> None:
    """Copy the icon, if one is supplied."""
    if self.config.applehelp_icon:
        try:
            with progress_message(__('copying icon... ')):
                applehelp_icon = path.join(self.srcdir, self.config.applehelp_icon)
                copy_asset_file(applehelp_icon, resources_dir)
        except Exception as err:
            logger.warning(__('cannot copy icon file %r: %s'), applehelp_icon, err)