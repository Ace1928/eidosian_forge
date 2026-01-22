from math import pi
from typing import Optional, Union
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array

        gate csx a,b { h b; cu1(pi/2) a,b; h b; }
        