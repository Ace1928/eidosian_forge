from functools import wraps
import inspect
from typing import Union, Callable, Tuple
import pennylane as qml
from .qnode import QNode, _make_execution_config, _get_device_shots
def get_transform_program(qnode: 'QNode', level=None) -> 'qml.transforms.core.TransformProgram':
    """Extract a transform program at a designated level.

    Args:
        qnode (QNode): the qnode to get the transform program for.
        level (None, str, int, slice): And indication of what transforms to use from the full program.

            * ``None``: use the full transform program
            * ``str``: Acceptable keys are ``"user"``, ``"device"``, ``"top"`` and ``"gradient"``
            * ``int``: How many transforms to include, starting from the front of the program
            * ``slice``: a slice to select out components of the transform program.

    Returns:
        TransformProgram: the transform program corresponding to the requested level.

    .. details::
        :title: Usage Details

        The transforms are organized as:

        .. image:: ../../_static/transforms_order.png
            :align: center
            :width: 800px
            :target: javascript:void(0);

        where ``transform1`` is first applied to the ``QNode`` followed by ``transform2``.  First user transforms are run on the tapes,
        followed by the gradient expansion, followed by the device expansion.  "Final" transforms, like ``param_shift`` and ``metric_tensor``,
        always occur at the end of the program.

        .. code-block:: python

            dev = qml.device('default.qubit')

            @qml.metric_tensor # final transform
            @qml.transforms.merge_rotations # transform 2
            @qml.transforms.cancel_inverses # transform 1
            @qml.qnode(dev, diff_method="parameter-shift", shifts=np.pi / 4)
            def circuit():
                return qml.expval(qml.Z(0))

        By default, we get the full transform program. This can be manually specified by ``level=None``.

        >>> qml.workflow.get_transform_program(circuit)
        TransformProgram(cancel_inverses, merge_rotations, _expand_metric_tensor,
        _expand_transform_param_shift, validate_device_wires, defer_measurements,
        decompose, validate_measurements, validate_observables, metric_tensor)

        The ``"user"`` transforms are the ones manually applied to the qnode, :class:`~.cancel_inverses` and
        :class:`~.merge_rotations`.

        >>> qml.workflow.get_transform_program(circuit, level="user")
        TransformProgram(cancel_inverses, merge_rotations)

        The ``_expand_transform_param_shift`` is the ``"gradient"`` transform.  This expands all trainable
        operations to a state where the parameter shift transform can operate on them. For example, it will decompose
        any parametrized templates into operators that have generators.

        >>> qml.workflow.get_transform_program(circuit, level="gradient")
        TransformProgram(cancel_inverses, merge_rotations, _expand_transform_param_shift)

        ``"device"`` includes all transforms except for a ``"final"`` transform, if it exists.  This usually
        corresponds to the circuits that will be sent to the device to execute.

        >>> qml.workflow.get_transform_program(circuit, level="device")
        TransformProgram(cancel_inverses, merge_rotations, _expand_transform_param_shift,
        validate_device_wires, defer_measurements, decompose, validate_measurements,
        validate_observables)

        ``"top"`` and ``0`` both return empty transform programs.

        >>> qml.workflow.get_transform_program(circuit, level="top")
        TransformProgram()
        >>> qml.workflow.get_transform_program(circuit, level=0)
        TransformProgram()

        The ``level`` can also be any integer, corresponding to a number of transforms in the program.

        >>> qml.workflow.get_transform_program(circuit, level=2)
        TransformProgram(cancel_inverses, merge_rotations)

        ``level`` can also accept a ``slice`` object to select out any arbitrary subset of the
        transform program.  This allows you to select different starting transforms or strides.
        For example, you can skip the first transform or reverse the order:

        >>> qml.workflow.get_transform_program(circuit, level=slice(1,3))
        TransformProgram(merge_rotations, _expand_transform_param_shift)
        >>> qml.workflow.get_transform_program(circuit, level=slice(None, None, -1))
        TransformProgram(metric_tensor, validate_observables, validate_measurements,
        decompose, defer_measurements, validate_device_wires, _expand_transform_param_shift,
        _expand_metric_tensor, merge_rotations, cancel_inverses)

    """
    full_transform_program = _get_full_transform_program(qnode)
    num_user = len(qnode.transform_program)
    if qnode.transform_program.has_final_transform:
        num_user -= 1
    if level == 'device':
        level = -1 if full_transform_program.has_final_transform else None
    elif level == 'top':
        level = 0
    elif level == 'user':
        level = num_user
    elif level == 'gradient':
        if getattr(qnode.gradient_fn, 'expand_transform', False):
            level = slice(0, num_user + 1)
        else:
            level = slice(0, num_user)
    elif isinstance(level, str):
        raise ValueError(f"level {level} not recognized. Acceptable strings are 'device', 'top', 'user', and 'gradient'.")
    if level is None or isinstance(level, int):
        level = slice(0, level)
    return full_transform_program[level]