import os
import textwrap
from io import StringIO
from sympy import __version__ as sympy_version
from sympy.core import Symbol, S, Tuple, Equality, Function, Basic
from sympy.printing.c import c_code_printers
from sympy.printing.codeprinter import AssignmentError
from sympy.printing.fortran import FCodePrinter
from sympy.printing.julia import JuliaCodePrinter
from sympy.printing.octave import OctaveCodePrinter
from sympy.printing.rust import RustCodePrinter
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
from sympy.utilities.iterables import is_sequence
def get_prototype(self, routine):
    """Returns a string for the function prototype of the routine.

        If the routine has multiple result objects, an CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
    results = [i.get_datatype('Rust') for i in routine.results]
    if len(results) == 1:
        rstype = ' -> ' + results[0]
    elif len(routine.results) > 1:
        rstype = ' -> (' + ', '.join(results) + ')'
    else:
        rstype = ''
    type_args = []
    for arg in routine.arguments:
        name = self.printer.doprint(arg.name)
        if arg.dimensions or isinstance(arg, ResultBase):
            type_args.append(('*%s' % name, arg.get_datatype('Rust')))
        else:
            type_args.append((name, arg.get_datatype('Rust')))
    arguments = ', '.join(['%s: %s' % t for t in type_args])
    return 'fn %s(%s)%s' % (routine.name, arguments, rstype)