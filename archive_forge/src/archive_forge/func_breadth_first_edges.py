import networkx as nx
from collections import deque
def breadth_first_edges(self, source, forbidden=set(), for_flow=False):
    """
        Generator for non-loop edges of a graph in the complement of
        a forbidden set, ordered by distance from the source.  Used for
        generating paths in the Ford-Fulkerson method.

        Yields triples (e, v, f) where e and f are edges containing v
        and e precedes f in the breadth-first ordering.

        The optional flag "for_flow" specifies that the flow_incident
        method should be used in place of the incident method for
        extending paths.  For example, Digraphs set flow_incident =
        outgoing.
        """
    incident = self.flow_incident if for_flow else self.incident
    initial = incident(source) - forbidden
    seen = forbidden | initial
    fifo = deque([(None, source, e) for e in initial])
    while fifo:
        parent, vertex, child = fifo.popleft()
        new_edges = incident(child(vertex)) - seen
        seen |= new_edges
        fifo.extend([(child, child(vertex), edge) for edge in new_edges])
        yield (parent, vertex, child)