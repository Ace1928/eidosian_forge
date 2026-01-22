import calendar
import re
import time
from . import osutils
Parse a patch-style date into a POSIX timestamp and offset.

    Inverse of format_patch_date.
    