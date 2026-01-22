from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import errno
import os
import pdb
import sys
import textwrap
import traceback
from absl import command_name
from absl import flags
from absl import logging
def define_help_flags():
    """Registers help flags. Idempotent."""
    global _define_help_flags_called
    if not _define_help_flags_called:
        flags.DEFINE_flag(HelpFlag())
        flags.DEFINE_flag(HelpshortFlag())
        flags.DEFINE_flag(HelpfullFlag())
        flags.DEFINE_flag(HelpXMLFlag())
        _define_help_flags_called = True