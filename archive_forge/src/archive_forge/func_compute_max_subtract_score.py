import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
def compute_max_subtract_score(self, column_index, cycle_indexes):
    """
        When updating scores the score of the highest-weighted incoming
        arc is subtracted upon collapse.  This returns the correct
        amount to subtract from that edge.

        :type column_index: integer.
        :param column_index: A index representing the column of incoming arcs
            to a particular node being updated
        :type cycle_indexes: A list of integers.
        :param cycle_indexes: Only arcs from cycle nodes are considered.  This
            is a list of such nodes addresses.
        """
    max_score = -100000
    for row_index in cycle_indexes:
        for subtract_val in self.scores[row_index][column_index]:
            if subtract_val > max_score:
                max_score = subtract_val
    return max_score