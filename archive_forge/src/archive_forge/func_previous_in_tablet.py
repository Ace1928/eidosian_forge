from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil
def previous_in_tablet(self, j):
    """
        :return: The position of the previous word that is in the same
            tablet as ``j``, or None if ``j`` is the first word of the
            tablet
        """
    i = self.alignment[j]
    tablet_position = self.cepts[i].index(j)
    if tablet_position == 0:
        return None
    return self.cepts[i][tablet_position - 1]