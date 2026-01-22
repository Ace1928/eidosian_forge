from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def split_label_fuzzy(self, label):
    """
        Splits a label entered as user input.

        It's more flexible in it's syntax parsing than the L{split_label_strict}
        method, as it allows the exclamation mark (B{C{!}}) to be omitted. The
        ambiguity is resolved by searching the modules in the snapshot to guess
        if a label refers to a module or a function. It also tries to rebuild
        labels when they contain hardcoded addresses.

        @warning: This method only parses the label, it doesn't make sure the
            label actually points to a valid memory location.

        @type  label: str
        @param label: Label to split.

        @rtype:  tuple( str or None, str or int or None, int or None )
        @return: Tuple containing the C{module} name,
            the C{function} name or ordinal, and the C{offset} value.

            If the label doesn't specify a module,
            then C{module} is C{None}.

            If the label doesn't specify a function,
            then C{function} is C{None}.

            If the label doesn't specify an offset,
            then C{offset} is C{0}.

        @raise ValueError: The label is malformed.
        """
    module = function = None
    offset = 0
    if not label:
        label = compat.b('0x0')
    else:
        label = label.replace(compat.b(' '), compat.b(''))
        label = label.replace(compat.b('\t'), compat.b(''))
        label = label.replace(compat.b('\r'), compat.b(''))
        label = label.replace(compat.b('\n'), compat.b(''))
        if not label:
            label = compat.b('0x0')
    if compat.b('!') in label:
        return self.split_label_strict(label)
    if compat.b('+') in label:
        try:
            prefix, offset = label.split(compat.b('+'))
        except ValueError:
            raise ValueError('Malformed label: %s' % label)
        try:
            offset = HexInput.integer(offset)
        except ValueError:
            raise ValueError('Malformed label: %s' % label)
        label = prefix
    modobj = self.get_module_by_name(label)
    if modobj:
        module = modobj.get_name()
    else:
        try:
            address = HexInput.integer(label)
            if offset:
                offset = address + offset
            else:
                offset = address
            try:
                new_label = self.get_label_at_address(offset)
                module, function, offset = self.split_label_strict(new_label)
            except ValueError:
                pass
        except ValueError:
            function = label
    if function and function.startswith(compat.b('#')):
        try:
            function = HexInput.integer(function[1:])
        except ValueError:
            pass
    if not offset:
        offset = None
    return (module, function, offset)