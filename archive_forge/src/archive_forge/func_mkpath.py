import sys
import os
import re
import logging
from .errors import DistutilsOptionError
from . import util, dir_util, file_util, archive_util, _modified
from ._log import log
def mkpath(self, name, mode=511):
    dir_util.mkpath(name, mode, dry_run=self.dry_run)