import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def format_option(self, option):
    """code taken from Python's optparse.py"""
    if option.help:
        from .i18n import gettext
        option.help = gettext(option.help)
    return optparse.IndentedHelpFormatter.format_option(self, option)