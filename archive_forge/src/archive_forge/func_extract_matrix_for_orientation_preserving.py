from ...sage_helper import _within_sage
@staticmethod
def extract_matrix_for_orientation_preserving(m):
    """
        Always returns a SageMath matrix whether given a SageMath matrix or
        an :class:`ExtendedMatrix`.

        Raises exception if given an orientation reversing extended matrix.
        """
    if isinstance(m, ExtendedMatrix):
        if m.isOrientationReversing:
            raise ValueError('Expected orientation preserving ExtendedMatrix.')
        return m.matrix
    return m