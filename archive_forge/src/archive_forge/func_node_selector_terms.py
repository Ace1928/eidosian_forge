from pprint import pformat
from six import iteritems
import re
@node_selector_terms.setter
def node_selector_terms(self, node_selector_terms):
    """
        Sets the node_selector_terms of this V1NodeSelector.
        Required. A list of node selector terms. The terms are ORed.

        :param node_selector_terms: The node_selector_terms of this
        V1NodeSelector.
        :type: list[V1NodeSelectorTerm]
        """
    if node_selector_terms is None:
        raise ValueError('Invalid value for `node_selector_terms`, must not be `None`')
    self._node_selector_terms = node_selector_terms