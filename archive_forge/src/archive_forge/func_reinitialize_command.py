import sys
import os
import re
import logging
from .errors import DistutilsOptionError
from . import util, dir_util, file_util, archive_util, _modified
from ._log import log
def reinitialize_command(self, command, reinit_subcommands=0):
    return self.distribution.reinitialize_command(command, reinit_subcommands)