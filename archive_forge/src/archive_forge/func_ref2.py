import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
def ref2(var):
    return var['C'] * var.get('temperature', 273) * var.get('Sfreq', 120000000000.0 / 273) / var.get('Sref', 1.2) * math.exp((var.get('Href', 1000.0) - var.get('Hact', 5000.0)) / var.get('temperature', 273))