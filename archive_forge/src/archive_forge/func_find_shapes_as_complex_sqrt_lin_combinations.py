import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
@sage_method
def find_shapes_as_complex_sqrt_lin_combinations(M, prec, degree):
    """
    Given a manifold M, use snap (which uses LLL-algorithm) with the given
    decimal precision and maximal degree to find exact values for the shapes'
    real and imaginary part. Return the shapes as list of
    ComplexSqrtLinCombination's. Return None on failure.

    Example::

       sage: from snappy import Manifold
       sage: M=Manifold("m412")
       sage: find_shapes_as_complex_sqrt_lin_combinations(M, 200, 10)
       [ComplexSqrtLinCombination((1/2) * sqrt(1), (x - 1/2) * sqrt(1)), ComplexSqrtLinCombination((1/2) * sqrt(1), (x - 1/2) * sqrt(1)), ComplexSqrtLinCombination((1/2) * sqrt(1), (x - 1/2) * sqrt(1)), ComplexSqrtLinCombination((1/2) * sqrt(1), (x - 1/2) * sqrt(1)), ComplexSqrtLinCombination((1/2) * sqrt(1), (x - 1/2) * sqrt(1))]
    """
    complex_data = M.tetrahedra_field_gens().find_field(prec, degree)
    if not complex_data:
        return None
    complex_number_field, complex_root, exact_complex_shapes = complex_data
    real_result = field_containing_real_and_imaginary_part_of_number_field(complex_number_field)
    if not real_result:
        return None
    real_number_field, real_part, imag_part = real_result
    embed_cache = {}
    exact_complex_root = ComplexSqrtLinCombination(real_part, imag_part, embed_cache=embed_cache)
    return [eval_number_field_elt(exact_complex_shape, exact_complex_root) for exact_complex_shape in exact_complex_shapes]