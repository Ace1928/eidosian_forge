from __future__ import annotations
import typing
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
import numpy
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit import Instruction, Parameter, ParameterVector, ParameterExpression
from qiskit.exceptions import QiskitError
from ..blueprintcircuit import BlueprintCircuit
def print_settings(self) -> str:
    """Returns information about the setting.

        Returns:
            The class name and the attributes/parameters of the instance as ``str``.
        """
    ret = f'NLocal: {self.__class__.__name__}\n'
    params = ''
    for key, value in self.__dict__.items():
        if key[0] == '_':
            params += f'-- {key[1:]}: {value}\n'
    ret += f'{params}'
    return ret