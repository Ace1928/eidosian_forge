from itertools import chain
from .exceptions import DMBadInputError, DMShapeError, DMDomainError
from .dense import (
from sympy.polys.domains import QQ
from .lll import ddm_lll, ddm_lll_transform
from .sdm import SDM
def flatiter(self):
    return chain.from_iterable(self)