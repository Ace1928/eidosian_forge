from typing import Any, Callable, Dict, Generic, Iterator, TypeVar, cast, TYPE_CHECKING
import functools
import networkx
from cirq import ops
from cirq.circuits import circuit
def get_root_node(some_node: Unique['cirq.Operation']) -> Unique['cirq.Operation']:
    pred = g.pred
    while pred[some_node]:
        some_node = next(iter(pred[some_node]))
    return some_node