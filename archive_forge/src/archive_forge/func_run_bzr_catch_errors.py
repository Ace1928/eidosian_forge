import contextlib
import os
import sys
from typing import List, Optional, Type, Union
from . import i18n, option, osutils, trace
from .lazy_import import lazy_import
import breezy
from breezy import (
from . import errors, registry
from .hooks import Hooks
from .i18n import gettext
from .plugin import disable_plugins, load_plugins, plugin_name
def run_bzr_catch_errors(argv):
    """Run a bzr command with parameters as described by argv.

    This function assumed that that UI layer is setup, that symbol deprecations
    are already applied, and that unicode decoding has already been performed
    on argv.
    """
    install_bzr_command_hooks()
    return exception_to_return_code(run_bzr, argv)