from snappy.verify.complex_volume.adjust_torsion import (
from snappy.verify.complex_volume.closed import zero_lifted_holonomy
from snappy.dev.extended_ptolemy import extended
from snappy.dev.extended_ptolemy import giac_rur
import snappy.snap.t3mlite as t3m
from sage.all import (RealIntervalField, ComplexIntervalField,
import sage.all
import re
def compute_representative_ptolemys_and_full_var_dict(M, precision=53):
    """
    Given a closed manifold (as Dehn-filling on 1-cusped manifold), compute
    a list of list of dictionaries assigning complex intervals to
    a subset of ptolemy variables and the full var dictionary to expand these
    to all variables.
    (Outer list for components, inner list for Galois conjugates).
    """
    I, full_var_dict = extended.ptolemy_ideal_for_filled(M, return_full_var_dict='data', notation='full')
    rur = giac_rur.rational_univariate_representation(I)
    return ([evaluate_at_roots(numberField, exact_values, precision) for numberField, exact_values, mult in rur], full_var_dict)