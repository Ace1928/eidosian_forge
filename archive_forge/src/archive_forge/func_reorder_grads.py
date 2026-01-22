from functools import partial
import warnings
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.measurements import (
def reorder_grads(grads, tape_specs):
    """Reorder the axes of tape gradients according to the original tape specifications.

    Args:
        grads (list[tensorlike] or list[tuple[tensorlike]] or list[tuple[tuple[tensorlike]]]:
            Gradient entries with leading parameter axis to be reordered.
        tape_specs (tuple): Information about the differentiated original tape in the order
            ``(bool: single_measure, int: num_params, int: num_measurements, Shots: shots)``.

    Returns:
        tensor_like or tuple[tensor_like] or tuple[tuple[tensor_like]]: The reordered gradient
            entries. Consider the details below for the ordering of the axes.

    The order of axes of the gradient output matches the structure outputted by jax.jacobian for
    a tuple-valued function. Internally, this may not be the case when computing the gradients,
    so the axes are reordered here.

    The axes of the input are assumed to be in the following order:

        1. Number of trainable parameters (Num params)
        2. Shot vector (if ``shots`` is a ``list`` or ``list[tuple]``. Skipped otherwise)
        3. Measurements (if there are multiple measurements. Skipped otherwise)
        4. Measurement shape
        5. Broadcasting dimension (for broadcasted tapes, skipped otherwise)

    The final order of axes of gradient results should be:

        1. Shot vector [1]
        2. Measurements [1]
        3. Number of trainable parameters (Num params) [1]
        4. Broadcasting dimension [2]
        5. Measurement shape

    [1] These axes are skipped in the output if they have length one. For shot vector and
        measurements, this already is true for the input. For num params, the axis is skipped
        "in addition", compared to the input.
    [2] Parameter broadcasting doesn't yet support multiple measurements, hence such cases are not
        dealt with at the moment by this function.

    The above reordering requires the following operations:

        1. In all cases, remove the parameter axis if it has length one.
        2. For a single measurement and no shot vector: Do nothing (but cast to ``tuple``)
        3. For a single measurement and shot vector: Swap first two axes (shots and parameters)
        4. For multiple measurements and no shot vector: Swap first two axes
           (measurements and parameters)
        5. For multiple measurements and shot vector: Move parameter axis from first to third
           position.

    In all cases the output will be a ``tuple``, except for single-measurement, single-parameter
    tapes, which will return a single measurement-like shaped output (no shot vector), or a list
    thereof (shot vector).
    """
    single_measure, num_params, num_measurements, shots = tape_specs
    if single_measure:
        if num_params == 1:
            return grads[0]
        if not shots.has_partitioned_shots:
            return tuple(grads)
        return _swap_first_two_axes(grads, num_params, shots.num_copies)
    if not shots.has_partitioned_shots:
        return _swap_first_two_axes(grads, num_params, num_measurements)
    return _move_first_axis_to_third_pos(grads, num_params, shots.num_copies, num_measurements)