from typing import Sequence, Any, Union, Dict
import numpy as np
import networkx as nx
import cirq
from cirq import GridQubit, LineQubit
from cirq.ops import NamedQubit
from cirq_pasqal import ThreeDQubit, TwoDQubit, PasqalGateset
@property
def supported_qubit_type(self):
    return (ThreeDQubit, TwoDQubit, GridQubit, LineQubit)