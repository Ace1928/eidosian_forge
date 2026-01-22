import pytest
import sys
from mpmath import *
def test_matrix_conversion_to_iv():
    for other_type_eye in [eye(3), fp.eye(3), iv.eye(3)]:
        A = iv.matrix(other_type_eye)
        B = iv.eye(3)
        assert type(A[0, 0]) == type(B[0, 0])
        assert A.tolist() == B.tolist()