from typing import Sequence, Callable
import itertools
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import (
from .finite_difference import finite_diff
from .general_shift_rules import generate_shifted_tapes, process_shifts
from .gradient_transform import _no_trainable_grad
from .parameter_shift import _get_operation_recipe, expval_param_shift
def second_order_param_shift(tape, dev_wires, argnum=None, shifts=None, gradient_recipes=None):
    """Generate the second-order CV parameter-shift tapes and postprocessing methods required
    to compute the gradient of a gate parameter with respect to an
    expectation value.

    .. note::

        The 2nd order method can handle also first-order observables, but
        1st order method may be more efficient unless it's really easy to
        experimentally measure arbitrary 2nd order observables.

    .. warning::

        The 2nd order method can only be executed on devices that support the
        :class:`~.PolyXP` observable.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        dev_wires (.Wires): wires on the device the parameter-shift method is computed on
        argnum (int or list[int] or None): Trainable parameter indices to differentiate
            with respect to. If not provided, the derivative with respect to all
            trainable indices are returned.
        shifts (list[tuple[int or float]]): List containing tuples of shift values.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple should match the number of frequencies for that parameter.
            If unspecified, equidistant shifts are assumed.
        gradient_recipes (tuple(list[list[float]] or None)): List of gradient recipes
            for the parameter-shift method. One gradient recipe must be provided
            per trainable parameter.

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing a
        list of generated tapes, together with a post-processing
        function to be applied to the results of the evaluated tapes
        in order to obtain the Jacobian matrix.
    """
    argnum = argnum or list(tape.trainable_params)
    gradient_recipes = gradient_recipes or [None] * len(argnum)
    gradient_tapes = []
    shapes = []
    obs_indices = []
    gradient_values = []
    for idx, _ in enumerate(tape.trainable_params):
        t_idx = list(tape.trainable_params)[idx]
        op = tape._par_info[t_idx]['op']
        if idx not in argnum:
            shapes.append(0)
            obs_indices.append([])
            gradient_values.append([])
            continue
        shapes.append(1)
        arg_idx = argnum.index(idx)
        recipe = gradient_recipes[arg_idx]
        if recipe is not None:
            recipe = process_shifts(np.array(recipe))
        else:
            op_shifts = None if shifts is None else shifts[arg_idx]
            recipe = _get_operation_recipe(tape, idx, shifts=op_shifts)
        coeffs, multipliers, op_shifts = recipe.T
        if len(op_shifts) != 2:
            raise NotImplementedError(f'Taking the analytic gradient for order-2 operators is unsupported for operation {op} which has a gradient recipe of more than two terms.')
        shifted_tapes = generate_shifted_tapes(tape, idx, op_shifts, multipliers)
        Z0 = op.heisenberg_tr(dev_wires, inverse=True)
        Z2 = shifted_tapes[0]._par_info[t_idx]['op'].heisenberg_tr(dev_wires)
        Z1 = shifted_tapes[1]._par_info[t_idx]['op'].heisenberg_tr(dev_wires)
        Z = Z2 * coeffs[0] + Z1 * coeffs[1]
        Z = Z @ Z0
        B = np.eye(1 + 2 * len(dev_wires))
        B_inv = B.copy()
        succ = tape.graph.descendants_in_order((op,))
        operation_descendents = itertools.filterfalse(qml.circuit_graph._is_observable, succ)
        observable_descendents = filter(qml.circuit_graph._is_observable, succ)
        for BB in operation_descendents:
            if not BB.supports_heisenberg:
                continue
            B = BB.heisenberg_tr(dev_wires) @ B
            B_inv = B_inv @ BB.heisenberg_tr(dev_wires, inverse=True)
        Z = B @ Z @ B_inv
        g_tape = tape.copy(copy_operations=True)
        constants = []
        transformed_obs_idx = []
        for mp in observable_descendents:
            obs = mp if mp.obs is None else mp.obs
            for obs_idx, tape_obs in enumerate(tape.observables):
                if obs is tape_obs:
                    break
            transformed_obs_idx.append(obs_idx)
            transformed_obs = _transform_observable(obs, Z, dev_wires)
            A = transformed_obs.parameters[0]
            constant = None
            if len(A.nonzero()[0]) == 1:
                if A.ndim == 2 and A[0, 0] != 0:
                    constant = A[0, 0]
                elif A.ndim == 1 and A[0] != 0:
                    constant = A[0]
            constants.append(constant)
            g_tape._measurements[obs_idx] = qml.expval(op=_transform_observable(obs, Z, dev_wires))
        g_tape._update_par_info()
        if not any((i is None for i in constants)):
            shapes[-1] = 0
            obs_indices.append(transformed_obs_idx)
            gradient_values.append(constants)
            continue
        gradient_tapes.append(g_tape)
        obs_indices.append(transformed_obs_idx)
        gradient_values.append(None)

    def processing_fn(results):
        grads = []
        start = 0
        if not results:
            results = [np.squeeze(np.zeros([tape.output_dim]))]
        interface = qml.math.get_interface(results[0])
        iterator = enumerate(zip(shapes, gradient_values, obs_indices))
        for i, (shape, grad_value, obs_ind) in iterator:
            if shape == 0:
                isscalar = qml.math.ndim(results[0]) == 0
                g = qml.math.zeros_like(qml.math.atleast_1d(results[0]), like=interface)
                if grad_value:
                    g = qml.math.scatter_element_add(g, obs_ind, grad_value, like=interface)
                grads.append(g[0] if isscalar else g)
                continue
            obs_result = results[start:start + shape]
            start = start + shape
            isscalar = qml.math.ndim(obs_result[0]) == 0
            obs_result = qml.math.stack(qml.math.atleast_1d(obs_result[0]))
            g = qml.math.zeros_like(obs_result, like=interface)
            if qml.math.get_interface(g) not in ('tensorflow', 'autograd'):
                obs_ind = (obs_ind,)
            g = qml.math.scatter_element_add(g, obs_ind, obs_result[obs_ind], like=interface)
            grads.append(g[0] if isscalar else g)
        for i, g in enumerate(grads):
            g = qml.math.convert_like(g, results[0])
            if hasattr(g, 'dtype') and g.dtype is np.dtype('object'):
                grads[i] = qml.math.hstack(g)
        return qml.math.T(qml.math.stack(grads))
    return (gradient_tapes, processing_fn)