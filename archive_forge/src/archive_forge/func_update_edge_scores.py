import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
def update_edge_scores(self, new_node, cycle_path):
    """
        Updates the edge scores to reflect a collapse operation into
        new_node.

        :type new_node: A Node.
        :param new_node: The node which cycle nodes are collapsed into.
        :type cycle_path: A list of integers.
        :param cycle_path: A list of node addresses that belong to the cycle.
        """
    logger.debug('cycle %s', cycle_path)
    cycle_path = self.compute_original_indexes(cycle_path)
    logger.debug('old cycle %s', cycle_path)
    logger.debug('Prior to update: %s', self.scores)
    for i, row in enumerate(self.scores):
        for j, column in enumerate(self.scores[i]):
            logger.debug(self.scores[i][j])
            if j in cycle_path and i not in cycle_path and self.scores[i][j]:
                subtract_val = self.compute_max_subtract_score(j, cycle_path)
                logger.debug('%s - %s', self.scores[i][j], subtract_val)
                new_vals = []
                for cur_val in self.scores[i][j]:
                    new_vals.append(cur_val - subtract_val)
                self.scores[i][j] = new_vals
    for i, row in enumerate(self.scores):
        for j, cell in enumerate(self.scores[i]):
            if i in cycle_path and j in cycle_path:
                self.scores[i][j] = []
    logger.debug('After update: %s', self.scores)