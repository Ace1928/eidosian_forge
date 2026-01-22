from sympy.polys.agca.modules import FreeModulePolyRing
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.domains.old_fractionfield import FractionField
from sympy.polys.domains.ring import Ring
from sympy.polys.orderings import monomial_key, build_product_order
from sympy.polys.polyclasses import DMP, DMF
from sympy.polys.polyerrors import (GeneratorsNeeded, PolynomialError,
from sympy.polys.polyutils import dict_from_basic, basic_from_dict, _dict_reorder
from sympy.utilities import public
from sympy.utilities.iterables import iterable
def from_GlobalPolynomialRing(K1, a, K0):
    """Convert a ``DMP`` object to ``dtype``. """
    if K1.gens == K0.gens:
        if K1.dom == K0.dom:
            return K1(a.rep)
        else:
            return K1(a.convert(K1.dom).rep)
    else:
        monoms, coeffs = _dict_reorder(a.to_dict(), K0.gens, K1.gens)
        if K1.dom != K0.dom:
            coeffs = [K1.dom.convert(c, K0.dom) for c in coeffs]
        return K1(dict(zip(monoms, coeffs)))