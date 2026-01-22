from sympy.polys.domains.field import Field
from sympy.polys.domains.modularinteger import ModularIntegerFactory
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
from sympy.polys.domains.groundtypes import SymPyInteger
def characteristic(self):
    """Return the characteristic of this domain. """
    return self.mod