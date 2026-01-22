import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring

    Example from Atkins, De Paula, Physical Chemistry
    H2 + Br2 -> 2HBr
    ½ dHBr/dt = k[H2][Br2]**(3/2) / ([Br2] + k'[HBr])
    