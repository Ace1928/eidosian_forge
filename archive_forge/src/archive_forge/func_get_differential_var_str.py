from importlib.metadata import version
import sympy
from sympy.external import import_module
from sympy.printing.str import StrPrinter
from sympy.physics.quantum.state import Bra, Ket
from .errors import LaTeXParsingError
def get_differential_var_str(text):
    for i in range(1, len(text)):
        c = text[i]
        if not (c == ' ' or c == '\r' or c == '\n' or (c == '\t')):
            idx = i
            break
    text = text[idx:]
    if text[0] == '\\':
        text = text[1:]
    return text