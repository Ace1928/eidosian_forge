from importlib.metadata import version
import sympy
from sympy.external import import_module
from sympy.printing.str import StrPrinter
from sympy.physics.quantum.state import Bra, Ket
from .errors import LaTeXParsingError
def convert_floor(floor):
    val = convert_expr(floor.val)
    return sympy.floor(val, evaluate=False)