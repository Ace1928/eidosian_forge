import sys
import os
import re
import logging
from .errors import DistutilsOptionError
from . import util, dir_util, file_util, archive_util, _modified
from ._log import log
def get_finalized_command(self, command, create=1):
    """Wrapper around Distribution's 'get_command_obj()' method: find
        (create if necessary and 'create' is true) the command object for
        'command', call its 'ensure_finalized()' method, and return the
        finalized command object.
        """
    cmd_obj = self.distribution.get_command_obj(command, create)
    cmd_obj.ensure_finalized()
    return cmd_obj