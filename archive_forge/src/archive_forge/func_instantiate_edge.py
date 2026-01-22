from time import perf_counter
from nltk.featstruct import TYPE, FeatStruct, find_variables, unify
from nltk.grammar import (
from nltk.parse.chart import (
from nltk.sem import logic
from nltk.tree import Tree
def instantiate_edge(self, edge):
    """
        If the edge is a ``FeatureTreeEdge``, and it is complete,
        then instantiate all variables whose names start with '@',
        by replacing them with unique new variables.

        Note that instantiation is done in-place, since the
        parsing algorithms might already hold a reference to
        the edge for future use.
        """
    if not isinstance(edge, FeatureTreeEdge):
        return
    if not edge.is_complete():
        return
    if edge in self._edge_to_cpls:
        return
    inst_vars = self.inst_vars(edge)
    if not inst_vars:
        return
    self._instantiated.add(edge)
    edge._lhs = edge.lhs().substitute_bindings(inst_vars)