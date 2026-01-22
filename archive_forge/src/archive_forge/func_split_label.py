from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
@classmethod
def split_label(cls, label):
    """
Splits a label into it's C{module}, C{function} and C{offset}
components, as used in L{parse_label}.

When called as a static method, the strict syntax mode is used::

    winappdbg.Process.split_label( "kernel32!CreateFileA" )

When called as an instance method, the fuzzy syntax mode is used::

    aProcessInstance.split_label( "CreateFileA" )

@see: L{split_label_strict}, L{split_label_fuzzy}

@type  label: str
@param label: Label to split.

@rtype:  tuple( str or None, str or int or None, int or None )
@return:
    Tuple containing the C{module} name,
    the C{function} name or ordinal, and the C{offset} value.

    If the label doesn't specify a module,
    then C{module} is C{None}.

    If the label doesn't specify a function,
    then C{function} is C{None}.

    If the label doesn't specify an offset,
    then C{offset} is C{0}.

@raise ValueError: The label is malformed.
        """
    return cls.split_label_strict(label)