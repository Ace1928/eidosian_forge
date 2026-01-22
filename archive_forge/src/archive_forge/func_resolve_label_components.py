from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def resolve_label_components(self, module=None, function=None, offset=None):
    """
        Resolve the memory address of the given module, function and/or offset.

        @note:
            If multiple modules with the same name are loaded,
            the label may be resolved at any of them. For a more precise
            way to resolve functions use the base address to get the L{Module}
            object (see L{Process.get_module}) and then call L{Module.resolve}.

            If no module name is specified in the label, the function may be
            resolved in any loaded module. If you want to resolve all functions
            with that name in all processes, call L{Process.iter_modules} to
            iterate through all loaded modules, and then try to resolve the
            function in each one of them using L{Module.resolve}.

        @type  module: None or str
        @param module: (Optional) Module name.

        @type  function: None, str or int
        @param function: (Optional) Function name or ordinal.

        @type  offset: None or int
        @param offset: (Optional) Offset value.

            If C{function} is specified, offset from the function.

            If C{function} is C{None}, offset from the module.

        @rtype:  int
        @return: Memory address pointed to by the label.

        @raise ValueError: The label is malformed or impossible to resolve.
        @raise RuntimeError: Cannot resolve the module or function.
        """
    address = 0
    if module:
        modobj = self.get_module_by_name(module)
        if not modobj:
            if module == 'main':
                modobj = self.get_main_module()
            else:
                raise RuntimeError('Module %r not found' % module)
        if function:
            address = modobj.resolve(function)
            if address is None:
                address = modobj.resolve_symbol(function)
                if address is None:
                    if function == 'start':
                        address = modobj.get_entry_point()
                    if address is None:
                        msg = 'Symbol %r not found in module %s'
                        raise RuntimeError(msg % (function, module))
        else:
            address = modobj.get_base()
    elif function:
        for modobj in self.iter_modules():
            address = modobj.resolve(function)
            if address is not None:
                break
        if address is None:
            if function == 'start':
                modobj = self.get_main_module()
                address = modobj.get_entry_point()
            elif function == 'main':
                modobj = self.get_main_module()
                address = modobj.get_base()
            else:
                msg = 'Function %r not found in any module' % function
                raise RuntimeError(msg)
    if offset:
        address = address + offset
    return address