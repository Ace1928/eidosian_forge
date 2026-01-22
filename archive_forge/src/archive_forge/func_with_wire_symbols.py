import re
from fractions import Fraction
from typing import (
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq import protocols, value
from cirq._doc import doc_private
def with_wire_symbols(self, new_wire_symbols: Iterable[str]):
    return CircuitDiagramInfo(wire_symbols=new_wire_symbols, exponent=self.exponent, connected=self.connected, exponent_qubit_index=self.exponent_qubit_index, auto_exponent_parens=self.auto_exponent_parens)