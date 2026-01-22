from dataclasses import replace
from functools import partial
from numbers import Number
from typing import Union, Callable, Tuple, Optional, Sequence
import concurrent.futures
import inspect
import logging
import numpy as np
import pennylane as qml
from pennylane.ops.op_math.condition import Conditional
from pennylane.measurements.mid_measure import MidMeasureMP
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from . import Device
from .modifiers import single_tape_support, simulator_tracking
from .preprocess import (
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .qubit.simulate import simulate, get_final_state, measure_final_state
from .qubit.adjoint_jacobian import adjoint_jacobian, adjoint_vjp, adjoint_jvp
def supports_derivatives(self, execution_config: Optional[ExecutionConfig]=None, circuit: Optional[QuantumTape]=None) -> bool:
    """Check whether or not derivatives are available for a given configuration and circuit.

        ``DefaultQubit`` supports backpropagation derivatives with analytic results, as well as
        adjoint differentiation.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information

        """
    if execution_config is None:
        return True
    if execution_config.gradient_method == 'backprop' and execution_config.device_options.get('max_workers', self._max_workers) is None and (execution_config.interface is not None):
        return True
    if execution_config.gradient_method == 'adjoint' and execution_config.use_device_gradient is not False:
        if circuit is None:
            return True
        prog = TransformProgram()
        _add_adjoint_transforms(prog)
        try:
            prog((circuit,))
        except (qml.operation.DecompositionUndefinedError, qml.DeviceError):
            return False
        return True
    return False