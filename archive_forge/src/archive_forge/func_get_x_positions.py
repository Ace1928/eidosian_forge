import math
import sys
from Bio import MissingPythonDependencyError
def get_x_positions(tree):
    """Create a mapping of each clade to its horizontal position.

        Dict of {clade: x-coord}
        """
    depths = tree.depths()
    if not max(depths.values()):
        depths = tree.depths(unit_branch_lengths=True)
    return depths