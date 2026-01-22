import abc
import dataclasses
from functools import lru_cache
from typing import Dict, Any, Tuple, List, Callable, TYPE_CHECKING, Hashable
import numpy as np
import cirq
from cirq import _compat
from cirq.devices.named_topologies import get_placements, NamedTopology
from cirq.protocols import obj_to_dict_helper
from cirq_google.workflow._device_shim import _Device_dot_get_nx_graph
def default_topo_node_to_qubit(node: Any) -> cirq.Qid:
    """The default mapping from `cirq.NamedTopology` nodes and `cirq.Qid`.

    There is a correspondence between nodes and the "abstract" Qids
    used to construct un-placed circuit. `cirq.get_placements` returns a dictionary
    mapping from node to Qid. We use this function to transform it into a mapping
    from "abstract" Qid to device Qid. This function encodes the default behavior used by
    `RandomDevicePlacer`.

    If nodes are tuples of integers, map to `cirq.GridQubit`. Otherwise, try
    to map to `cirq.LineQubit` and rely on its validation.

    Args:
        node: A node from a `cirq.NamedTopology` graph.

    Returns:
        A `cirq.Qid` appropriate for the node type.
    """
    try:
        return cirq.GridQubit(*node)
    except TypeError:
        return cirq.LineQubit(node)