from __future__ import annotations
import numpy as np
Given a symmetric matrix in upper triangular matrix form as flat array indexes as:
    [A_xx,A_yy,A_zz,A_xy,A_xz,A_yz]
    This will generate the full matrix:
    [[A_xx,A_xy,A_xz],[A_xy,A_yy,A_yz],[A_xz,A_yz,A_zz].
    