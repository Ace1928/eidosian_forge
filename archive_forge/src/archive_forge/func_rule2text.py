from importlib.metadata import version
import sympy
from sympy.external import import_module
from sympy.printing.str import StrPrinter
from sympy.physics.quantum.state import Bra, Ket
from .errors import LaTeXParsingError
def rule2text(ctx):
    stream = ctx.start.getInputStream()
    startIdx = ctx.start.start
    stopIdx = ctx.stop.stop
    return stream.getText(startIdx, stopIdx)