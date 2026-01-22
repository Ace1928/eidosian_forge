import numpy as np
from .layer import Layer1Q, Layer2Q

        Applies the left (row) and right (column) permutations to the matrix.
        at the end of computation process.

        Args:
            temp_mat: temporary, external matrix.

        Returns:
            finalized matrix with all transformations applied.
        