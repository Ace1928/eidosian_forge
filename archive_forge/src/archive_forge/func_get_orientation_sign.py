from ...sage_helper import _within_sage
@staticmethod
def get_orientation_sign(m):
    """
        Returns +1 or -1 depending on whether the given (extended) matrix
        acts orientation preserving or reversing.
        """
    if isinstance(m, ExtendedMatrix):
        if m.isOrientationReversing:
            return -1
    return +1