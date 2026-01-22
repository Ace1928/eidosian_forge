import warnings
from collections.abc import Sequence as SequenceType
from dataclasses import InitVar, dataclass
from typing import Any, ClassVar, Dict, List, Sequence, Union
from networkx import MultiDiGraph
import pennylane as qml
from pennylane.ops.meta import WireCut
def get_cut_kwargs(self, tape_dag: MultiDiGraph, max_wires_by_fragment: Sequence[int]=None, max_gates_by_fragment: Sequence[int]=None, exhaustive: bool=True) -> List[Dict[str, Any]]:
    """Derive the complete set of arguments, based on a given circuit, for passing to a graph
        partitioner.

        Args:
            tape_dag (nx.MultiDiGraph): Graph representing a tape, typically the output of
                :func:`tape_to_graph`.
            max_wires_by_fragment (Sequence[int]): User-predetermined list of wire limits by
                fragment. If supplied, the number of fragments will be derived from it and
                exploration of other choices will not be made.
            max_gates_by_fragment (Sequence[int]): User-predetermined list of gate limits by
                fragment. If supplied, the number of fragments will be derived from it and
                exploration of other choices will not be made.
            exhaustive (bool): Toggle for an exhaustive search which will attempt all potentially
                valid numbers of fragments into which the circuit is partitioned. If ``True``,
                for a circuit with N gates, N - 1 attempts will be made with ``num_fragments``
                ranging from [2, N], i.e. from bi-partitioning to complete partitioning where each
                fragment has exactly a single gate. Defaults to ``True``.

        Returns:
            List[Dict[str, Any]]: A list of minimal kwargs being passed to a graph
            partitioner method.

        **Example**

        Deriving kwargs for a given circuit and feeding them to a custom partitioner, along with
        extra parameters specified using ``extra_kwargs``:

        >>> cut_strategy = qcut.CutStrategy(devices=dev)
        >>> cut_kwargs = cut_strategy.get_cut_kwargs(tape_dag)
        >>> cut_trials = [
        ...     my_partition_fn(tape_dag, **kwargs, **extra_kwargs) for kwargs in cut_kwargs
        ... ]

        """
    wire_depths = {}
    for g in tape_dag.nodes:
        if not isinstance(g.obj, WireCut):
            for w in g.obj.wires:
                wire_depths[w] = wire_depths.get(w, 0) + 1 / len(g.obj.wires)
    self._validate_input(max_wires_by_fragment, max_gates_by_fragment)
    probed_cuts = self._infer_probed_cuts(wire_depths=wire_depths, max_wires_by_fragment=max_wires_by_fragment, max_gates_by_fragment=max_gates_by_fragment, exhaustive=exhaustive)
    return probed_cuts