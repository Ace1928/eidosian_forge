import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
def original_best_arc(self, node_index):
    originals = self.compute_original_indexes([node_index])
    max_arc = None
    max_score = None
    max_orig = None
    for row_index in range(len(self.scores)):
        for col_index in range(len(self.scores[row_index])):
            if col_index in originals and (max_score is None or self.scores[row_index][col_index] > max_score):
                max_score = self.scores[row_index][col_index]
                max_arc = row_index
                max_orig = col_index
    return [max_arc, max_orig]