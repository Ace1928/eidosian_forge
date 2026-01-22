from ...sage_helper import _within_sage
@staticmethod
def get_trace_in_PSL(m):
    """
        Given an (extended) matrix acting in an orientation preserving way,
        computes the trace after normalizing the matrix to be in SL(2,C).
        """
    m = ExtendedMatrix.extract_matrix_for_orientation_preserving(m)
    return (m[0, 0] + m[1, 1]) / sqrt(m.det())