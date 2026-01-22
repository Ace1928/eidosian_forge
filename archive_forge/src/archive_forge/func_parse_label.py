from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
@staticmethod
def parse_label(module=None, function=None, offset=None):
    """
        Creates a label from a module and a function name, plus an offset.

        @warning: This method only creates the label, it doesn't make sure the
            label actually points to a valid memory location.

        @type  module: None or str
        @param module: (Optional) Module name.

        @type  function: None, str or int
        @param function: (Optional) Function name or ordinal.

        @type  offset: None or int
        @param offset: (Optional) Offset value.

            If C{function} is specified, offset from the function.

            If C{function} is C{None}, offset from the module.

        @rtype:  str
        @return:
            Label representing the given function in the given module.

        @raise ValueError:
            The module or function name contain invalid characters.
        """
    try:
        function = '#0x%x' % function
    except TypeError:
        pass
    if module is not None and ('!' in module or '+' in module):
        raise ValueError('Invalid module name: %s' % module)
    if function is not None and ('!' in function or '+' in function):
        raise ValueError('Invalid function name: %s' % function)
    if module:
        if function:
            if offset:
                label = '%s!%s+0x%x' % (module, function, offset)
            else:
                label = '%s!%s' % (module, function)
        elif offset:
            label = '%s!0x%x' % (module, offset)
        else:
            label = '%s!' % module
    elif function:
        if offset:
            label = '!%s+0x%x' % (function, offset)
        else:
            label = '!%s' % function
    elif offset:
        label = '0x%x' % offset
    else:
        label = '0x0'
    return label