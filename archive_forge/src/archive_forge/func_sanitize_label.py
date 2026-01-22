from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def sanitize_label(self, label):
    """
        Converts a label taken from user input into a well-formed label.

        @type  label: str
        @param label: Label taken from user input.

        @rtype:  str
        @return: Sanitized label.
        """
    module, function, offset = self.split_label_fuzzy(label)
    label = self.parse_label(module, function, offset)
    return label