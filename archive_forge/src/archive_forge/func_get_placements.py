import abc
import warnings
from dataclasses import dataclass
from typing import (
import networkx as nx
from matplotlib import pyplot as plt
from cirq import _compat
from cirq.devices import GridQubit, LineQubit
from cirq.protocols.json_serialization import dataclass_json_dict
def get_placements(big_graph: nx.Graph, small_graph: nx.Graph, max_placements=100000) -> List[Dict]:
    """Get 'placements' mapping small_graph nodes onto those of `big_graph`.

    This function considers monomorphisms with a restriction: we restrict only to unique set
    of `big_graph` qubits. Some monomorphisms may be basically
    the same mapping just rotated/flipped which we purposefully exclude. This could
    exclude meaningful differences like using the same qubits but having the edges assigned
    differently, but it prevents the number of placements from blowing up.

    Args:
        big_graph: The parent, super-graph. We often consider the case where this is a
            nx.Graph representation of a Device whose nodes are `cirq.Qid`s like `GridQubit`s.
        small_graph: The subgraph. We often consider the case where this is a NamedTopology
            graph.
        max_placements: Raise a value error if there are more than this many placement
            possibilities. It is possible to use `big_graph`, `small_graph` combinations
            that result in an intractable number of placements.

    Raises:
        ValueError: if the number of placements exceeds `max_placements`.

    Returns:
        A list of placement dictionaries. Each dictionary maps the nodes in `small_graph` to
        nodes in `big_graph` with a monomorphic relationship. That's to say: if an edge exists
        in `small_graph` between two nodes, it will exist in `big_graph` between the mapped nodes.
    """
    matcher = nx.algorithms.isomorphism.GraphMatcher(big_graph, small_graph)
    dedupe = {}
    for big_to_small_map in matcher.subgraph_monomorphisms_iter():
        dedupe[frozenset(big_to_small_map.keys())] = big_to_small_map
        if len(dedupe) > max_placements:
            raise ValueError(f'We found more than {max_placements} placements. Please use a more constraining `big_graph` or a more constrained `small_graph`.')
    small_to_bigs = []
    for big in sorted(dedupe.keys()):
        big_to_small_map = dedupe[big]
        small_to_big_map = {v: k for k, v in big_to_small_map.items()}
        small_to_bigs.append(small_to_big_map)
    return small_to_bigs