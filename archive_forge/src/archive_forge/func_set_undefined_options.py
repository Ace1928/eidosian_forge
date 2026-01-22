import sys
import os
import re
import logging
from .errors import DistutilsOptionError
from . import util, dir_util, file_util, archive_util, _modified
from ._log import log
def set_undefined_options(self, src_cmd, *option_pairs):
    """Set the values of any "undefined" options from corresponding
        option values in some other command object.  "Undefined" here means
        "is None", which is the convention used to indicate that an option
        has not been changed between 'initialize_options()' and
        'finalize_options()'.  Usually called from 'finalize_options()' for
        options that depend on some other command rather than another
        option of the same command.  'src_cmd' is the other command from
        which option values will be taken (a command object will be created
        for it if necessary); the remaining arguments are
        '(src_option,dst_option)' tuples which mean "take the value of
        'src_option' in the 'src_cmd' command object, and copy it to
        'dst_option' in the current command object".
        """
    src_cmd_obj = self.distribution.get_command_obj(src_cmd)
    src_cmd_obj.ensure_finalized()
    for src_option, dst_option in option_pairs:
        if getattr(self, dst_option) is None:
            setattr(self, dst_option, getattr(src_cmd_obj, src_option))