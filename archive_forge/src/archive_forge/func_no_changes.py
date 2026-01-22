import contextlib
import shutil
import sys
import tempfile
from io import BytesIO
import patiencediff
from . import (builtins, delta, diff, errors, osutils, patches, shelf,
from .i18n import gettext
def no_changes(self):
    """Report that no changes were selected to apply."""
    trace.warning('No changes to shelve.')