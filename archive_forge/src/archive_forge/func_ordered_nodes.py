from typing import Any, Callable, Dict, Generic, Iterator, TypeVar, cast, TYPE_CHECKING
import functools
import networkx
from cirq import ops
from cirq.circuits import circuit
def ordered_nodes(self) -> Iterator[Unique['cirq.Operation']]:
    if not self.nodes():
        return
    g = self.copy()

    def get_root_node(some_node: Unique['cirq.Operation']) -> Unique['cirq.Operation']:
        pred = g.pred
        while pred[some_node]:
            some_node = next(iter(pred[some_node]))
        return some_node

    def get_first_node() -> Unique['cirq.Operation']:
        return get_root_node(next(iter(g.nodes())))

    def get_next_node(succ: networkx.classes.coreviews.AtlasView) -> Unique['cirq.Operation']:
        if succ:
            return get_root_node(next(iter(succ)))
        return get_first_node()
    node = get_first_node()
    while True:
        yield node
        succ = g.succ[node]
        g.remove_node(node)
        if not g.nodes():
            return
        node = get_next_node(succ)