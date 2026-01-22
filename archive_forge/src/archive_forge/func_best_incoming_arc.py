import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
def best_incoming_arc(self, node_index):
    """
        Returns the source of the best incoming arc to the
        node with address: node_index

        :type node_index: integer.
        :param node_index: The address of the 'destination' node,
            the node that is arced to.
        """
    originals = self.compute_original_indexes([node_index])
    logger.debug('originals: %s', originals)
    max_arc = None
    max_score = None
    for row_index in range(len(self.scores)):
        for col_index in range(len(self.scores[row_index])):
            if col_index in originals and (max_score is None or self.scores[row_index][col_index] > max_score):
                max_score = self.scores[row_index][col_index]
                max_arc = row_index
                logger.debug('%s, %s', row_index, col_index)
    logger.debug(max_score)
    for key in self.inner_nodes:
        replaced_nodes = self.inner_nodes[key]
        if max_arc in replaced_nodes:
            return key
    return max_arc