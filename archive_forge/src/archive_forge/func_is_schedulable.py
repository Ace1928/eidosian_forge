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
def is_schedulable(self) -> bool:
    """Return ``True`` if all durations are assigned."""
    for context_param in self._alignment_context._context_params:
        if isinstance(context_param, ParameterExpression):
            return False
    for elm in self.blocks:
        if isinstance(elm, ScheduleBlock):
            if not elm.is_schedulable():
                return False
        else:
            try:
                if not isinstance(elm.duration, int):
                    return False
            except UnassignedReferenceError:
                return False
    return True