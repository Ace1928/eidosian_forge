from typing import NamedTuple
from typing import Sequence, Tuple
@property
def has_partitioned_shots(self):
    """
        Evaluates to True if this instance represents either multiple shot
        quantities, or the same shot quantity repeated multiple times.

        Returns:
            bool: whether shots are partitioned
        """
    if not self:
        return False
    return len(self.shot_vector) > 1 or self.shot_vector[0].copies > 1