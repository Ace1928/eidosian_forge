from __future__ import annotations
import abc
import copy
import functools
import itertools
import multiprocessing as mp
import sys
import warnings
from collections.abc import Callable, Iterable
from typing import List, Tuple, Union, Dict, Any
import numpy as np
import rustworkx as rx
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError, UnassignedReferenceError
from qiskit.pulse.instructions import Instruction, Reference
from qiskit.pulse.utils import instruction_duration_validation
from qiskit.pulse.reference_manager import ReferenceManager
from qiskit.utils.multiprocessing import is_main_process
@classmethod
def initialize_from(cls, other_program: Any, name: str | None=None) -> 'ScheduleBlock':
    """Create new schedule object with metadata of another schedule object.

        Args:
            other_program: Qiskit program that provides metadata to new object.
            name: Name of new schedule. Name of ``block`` is used by default.

        Returns:
            New block object with name and metadata.

        Raises:
            PulseError: When ``other_program`` does not provide necessary information.
        """
    try:
        name = name or other_program.name
        if other_program.metadata:
            metadata = other_program.metadata.copy()
        else:
            metadata = None
        try:
            alignment_context = other_program.alignment_context
        except AttributeError:
            alignment_context = None
        return cls(name=name, metadata=metadata, alignment_context=alignment_context)
    except AttributeError as ex:
        raise PulseError(f'{cls.__name__} cannot be initialized from the program data {other_program.__class__.__name__}.') from ex