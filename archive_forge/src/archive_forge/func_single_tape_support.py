from functools import wraps
from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScript
def single_tape_support(cls: type) -> type:
    """Modifies all functions to accept single tapes in addition to batches. This allows the definition
    of the device class to purely focus on executing batches.

    Args:
        cls (type): a subclass of :class:`pennylane.devices.Device`

    Returns
        type: The inputted class that has now been modified to accept single circuits as well as batches.

    .. code-block:: python

        @single_tape_support
        class MyDevice(qml.devices.Device):

            def execute(self, circuits, execution_config = qml.devices.DefaultExecutionConfig):
                return tuple(0.0 for _ in circuits)

    >>> dev = MyDevice()
    >>> t = qml.tape.QuantumScript()
    >>> dev.execute(t)
    0.0
    >>> dev.execute((t, ))
    (0.0,)

    In this situation, ``MyDevice.execute`` only needs to handle the case where ``circuits`` is an iterable
    of :class:`~pennylane.tape.QuantumTape` objects, not a single value.

    """
    if not issubclass(cls, Device):
        raise ValueError('single_tape_support only accepts subclasses of pennylane.devices.Device')
    if hasattr(cls, '_applied_modifiers'):
        cls._applied_modifiers.append(single_tape_support)
    else:
        cls._applied_modifiers = [single_tape_support]
    cls.execute = _make_execute(cls.execute)
    modifier_map = {'compute_derivatives': _make_compute_derivatives, 'execute_and_compute_derivatives': _make_execute_and_compute_derivatives, 'compute_jvp': _make_compute_jvp, 'execute_and_compute_jvp': _make_execute_and_compute_jvp, 'compute_vjp': _make_compute_vjp, 'execute_and_compute_vjp': _make_execute_and_compute_vjp}
    for name, modifier in modifier_map.items():
        if getattr(cls, name) != getattr(Device, name):
            original = getattr(cls, name)
            setattr(cls, name, modifier(original))
    return cls