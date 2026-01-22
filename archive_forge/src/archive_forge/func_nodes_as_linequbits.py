import abc
import warnings
from dataclasses import dataclass
from typing import (
import networkx as nx
from matplotlib import pyplot as plt
from cirq import _compat
from cirq.devices import GridQubit, LineQubit
from cirq.protocols.json_serialization import dataclass_json_dict
def nodes_as_linequbits(self) -> List['cirq.LineQubit']:
    """Get the graph nodes as cirq.LineQubit"""
    return [LineQubit(x) for x in sorted(self.graph.nodes)]