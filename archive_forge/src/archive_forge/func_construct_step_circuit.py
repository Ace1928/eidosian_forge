import networkx as nx
import pytest
import cirq
def construct_step_circuit(k: int):
    q = cirq.LineQubit.range(k)
    return cirq.Circuit([cirq.CNOT(q[i], q[i + 1]) for i in range(k - 1)])