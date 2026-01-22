from breezy.lazy_import import lazy_import
from ... import config, merge
import fnmatch
import subprocess
import tempfile
from breezy import (
Calls msgmerge when .po files conflict.

        This requires a valid .pot file to reconcile both sides.
        