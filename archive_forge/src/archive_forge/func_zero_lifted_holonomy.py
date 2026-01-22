from ...sage_helper import _within_sage, sage_method
from ...math_basics import prod
from ...snap import peripheral
from .adjust_torsion import *
from .compute_ptolemys import *
from .. import verifyHyperbolicity
from ..cuspCrossSection import ComplexCuspCrossSection
from ...snap import t3mlite as t3m
@sage_method
def zero_lifted_holonomy(manifold, m, l, f):
    """
    Given a closed manifold and any log of the holonomy of the meridian and
    longitude, adjust logs by multiplies of f pi i such that the peripheral
    curves goes to 0.
    """
    CIF = m.parent()
    RIF = CIF.real_field()
    multiple_of_pi = RIF(f * pi)
    m_fill, l_fill = [int(x) for x in manifold.cusp_info()[0]['filling']]
    p_interval = (m_fill * m + l_fill * l).imag() / multiple_of_pi
    is_int, p = p_interval.is_int()
    if not is_int:
        raise Exception('Expected multiple of %d * pi * i (increase precision?)' % f)
    if p == 0:
        return (m, l)
    g, a, b = xgcd(m_fill, l_fill)
    m -= p * a * multiple_of_pi * sage.all.I
    l -= p * b * multiple_of_pi * sage.all.I
    p_interval = (m_fill * m + l_fill * l).imag() / multiple_of_pi
    is_int, p = p_interval.is_int()
    if not is_int:
        raise Exception('Expected multiple of %d * pi * i (increase precision?)' % f)
    if p != 0:
        raise Exception('Expected 0')
    return (m, l)