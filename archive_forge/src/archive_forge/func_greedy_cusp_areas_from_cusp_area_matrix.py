from ..sage_helper import _within_sage
from ..math_basics import correct_min, is_RealIntervalFieldElement
def greedy_cusp_areas_from_cusp_area_matrix(cusp_area_matrix, first_cusps=[]):
    """

        sage: from sage.all import matrix, RIF
        sage: greedy_cusp_areas_from_cusp_area_matrix(
        ...             matrix([[RIF(9.0,9.0005),RIF(6.0, 6.001)],
        ...                     [RIF(6.0,6.001 ),RIF(10.0, 10.001)]]))
        [3.0001?, 2.000?]

        >>> from snappy.SnapPy import matrix
        >>> greedy_cusp_areas_from_cusp_area_matrix(
        ...             matrix([[10.0, 40.0],
        ...                     [40.0, 20.0]]))
        [3.1622776601683795, 4.47213595499958]

    """
    num_cusps = cusp_area_matrix.dimensions()[0]
    result = list(range(num_cusps))
    sigma = first_cusps + [i for i in range(num_cusps) if i not in first_cusps]
    for i in range(num_cusps):
        stoppers = [cusp_area_matrix[sigma[i], sigma[j]] / result[sigma[j]] for j in range(i)]
        self_stopper = sqrt(cusp_area_matrix[sigma[i], sigma[i]])
        result[sigma[i]] = correct_min(stoppers + [self_stopper])
    return result