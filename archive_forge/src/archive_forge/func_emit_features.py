import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
def emit_features(self):
    for feature in sorted(commands.FEATURE_NAMES):
        self.print_cmd(commands.FeatureCommand(feature))