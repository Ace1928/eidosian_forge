import re
import string
def sgn_column(matrix, col):
    """
    Returns +1 or -1 depending on the sign of the first non-zero entry
    in the column of the given matrix.
    """
    first_non_zero_entry = matrix[0, col] if matrix[0, col] != 0 else matrix[1, col]
    return +1 if first_non_zero_entry > 0 else -1