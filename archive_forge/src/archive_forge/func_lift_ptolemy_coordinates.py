from snappy.verify.complex_volume.adjust_torsion import (
from snappy.verify.complex_volume.closed import zero_lifted_holonomy
from snappy.dev.extended_ptolemy import extended
from snappy.dev.extended_ptolemy import giac_rur
import snappy.snap.t3mlite as t3m
from sage.all import (RealIntervalField, ComplexIntervalField,
import sage.all
import re
def lift_ptolemy_coordinates(M, solution, full_var_dict):
    """
    Given a closed manifold (as Dehn-filling on 1-cusped manifold) and an
    assignment of subset of ptolemy variables and the full var dict, compute
    logs for all Ptolemy's.
    """
    lifted = {str(k): log(v) for k, v in solution.items() if str(k)[0].islower()}
    m, l = zero_lifted_holonomy(M, lifted['m'], lifted['l'], 2)
    return {k: lifted[name] - (m_count * m + l_count * l) for k, (sign, m_count, l_count, name) in full_var_dict.items() if k[0] == 'c'}