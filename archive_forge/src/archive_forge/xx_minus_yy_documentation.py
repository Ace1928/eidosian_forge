import math
from cmath import exp
from math import pi
from typing import Optional
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit.library.standard_gates.s import SdgGate, SGate
from qiskit.circuit.library.standard_gates.sx import SXdgGate, SXGate
from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
Raise gate to a power.