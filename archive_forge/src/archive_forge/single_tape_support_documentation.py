from functools import wraps
from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScript
Modifies all functions to accept single tapes in addition to batches. This allows the definition
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

    