import dataclasses
import inspect
import enum
import functools
import textwrap
from typing import (
from typing_extensions import Protocol
from cirq import circuits
def register_final(self, circuit: 'cirq.AbstractCircuit', transformer_name: str) -> None:
    pass