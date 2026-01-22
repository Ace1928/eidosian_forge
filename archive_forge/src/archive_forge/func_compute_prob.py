from collections import defaultdict
from functools import total_ordering
from itertools import chain
from nltk.grammar import (
from nltk.internals import raise_unorderable_types
from nltk.parse.dependencygraph import DependencyGraph
def compute_prob(self, dg):
    """
        Computes the probability of a dependency graph based
        on the parser's probability model (defined by the parser's
        statistical dependency grammar).

        :param dg: A dependency graph to score.
        :type dg: DependencyGraph
        :return: The probability of the dependency graph.
        :rtype: int
        """
    prob = 1.0
    for node_index in range(1, len(dg.nodes)):
        children = list(chain.from_iterable(dg.nodes[node_index]['deps'].values()))
        nr_left_children = dg.left_children(node_index)
        nr_right_children = dg.right_children(node_index)
        nr_children = nr_left_children + nr_right_children
        for child_index in range(0 - (nr_left_children + 1), nr_right_children + 2):
            head_word = dg.nodes[node_index]['word']
            head_tag = dg.nodes[node_index]['tag']
            child = 'STOP'
            child_tag = 'STOP'
            prev_word = 'START'
            prev_tag = 'START'
            if child_index < 0:
                array_index = child_index + nr_left_children
                if array_index >= 0:
                    child = dg.nodes[children[array_index]]['word']
                    child_tag = dg.nodes[children[array_index]]['tag']
                if child_index != -1:
                    prev_word = dg.nodes[children[array_index + 1]]['word']
                    prev_tag = dg.nodes[children[array_index + 1]]['tag']
                head_event = '(head ({} {}) (mods ({}, {}, {}) left))'.format(child, child_tag, prev_tag, head_word, head_tag)
                mod_event = '(mods ({}, {}, {}) left))'.format(prev_tag, head_word, head_tag)
                h_count = self._grammar._events[head_event]
                m_count = self._grammar._events[mod_event]
                if m_count != 0:
                    prob *= h_count / m_count
                else:
                    prob = 1e-08
            elif child_index > 0:
                array_index = child_index + nr_left_children - 1
                if array_index < nr_children:
                    child = dg.nodes[children[array_index]]['word']
                    child_tag = dg.nodes[children[array_index]]['tag']
                if child_index != 1:
                    prev_word = dg.nodes[children[array_index - 1]]['word']
                    prev_tag = dg.nodes[children[array_index - 1]]['tag']
                head_event = '(head ({} {}) (mods ({}, {}, {}) right))'.format(child, child_tag, prev_tag, head_word, head_tag)
                mod_event = '(mods ({}, {}, {}) right))'.format(prev_tag, head_word, head_tag)
                h_count = self._grammar._events[head_event]
                m_count = self._grammar._events[mod_event]
                if m_count != 0:
                    prob *= h_count / m_count
                else:
                    prob = 1e-08
    return prob