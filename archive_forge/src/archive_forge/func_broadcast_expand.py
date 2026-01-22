from typing import Sequence, Callable
import pennylane as qml
from .core import transform
@transform
def broadcast_expand(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Expand a broadcasted tape into multiple tapes
    and a function that stacks and squeezes the results.

    .. warning::

        Currently, not all templates have been updated to support broadcasting.

    Args:
        tape (QNode or QuantumTape or Callable): Broadcasted tape to be expanded

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

        - If the input is a QNode, the broadcasted input QNode
          that computes the QNode output serially with multiple circuit evaluations and
          stacks (and squeezes) the results into one batch of results.

        - If the input is a tape, a tuple containing a list of generated tapes, together with
          a post-processing function. The number of tapes matches the broadcasting dimension
          of the input tape, and the results from the evaluated tapes are stacked and squeezed
          together in the post-processing function.

    This expansion function is used internally whenever a device does not
    support broadcasting.

    **Example**

    We may use ``broadcast_expand`` on a ``QNode`` to separate it
    into multiple calculations. For this we will provide ``qml.RX`` with
    the ``ndim_params`` attribute that allows the operation to detect
    broadcasting, and set up a simple ``QNode`` with a single operation and
    returned expectation value:

    >>> qml.RX.ndim_params = (0,)
    >>> dev = qml.device("default.qubit", wires=1)
    >>> @qml.qnode(dev)
    >>> def circuit(x):
    ...     qml.RX(x, wires=0)
    ...     return qml.expval(qml.Z(0))

    We can then call ``broadcast_expand`` on the QNode and store the
    expanded ``QNode``:

    >>> expanded_circuit = qml.transforms.broadcast_expand(circuit)

    Let's use the expanded QNode and draw it for broadcasted parameters
    with broadcasting axis of length ``3`` passed to ``qml.RX``:

    >>> x = pnp.array([0.2, 0.6, 1.0], requires_grad=True)
    >>> print(qml.draw(expanded_circuit)(x))
    0: ──RX(0.20)─┤  <Z>
    0: ──RX(0.60)─┤  <Z>
    0: ──RX(1.00)─┤  <Z>

    Executing the expanded ``QNode`` results in three values, corresponding
    to the three parameters in the broadcasted input ``x``:

    >>> expanded_circuit(x)
    tensor([0.98006658, 0.82533561, 0.54030231], requires_grad=True)

    We also can call the transform manually on a tape:

    >>> ops = [qml.RX(pnp.array([0.2, 0.6, 1.0], requires_grad=True), wires=0)]
    >>> measurements = [qml.expval(qml.Z(0))]
    >>> tape = qml.tape.QuantumTape(ops, measurements)
    >>> tapes, fn = qml.transforms.broadcast_expand(tape)
    >>> tapes
    [<QuantumTape: wires=[0], params=1>, <QuantumTape: wires=[0], params=1>, <QuantumTape: wires=[0], params=1>]
    >>> fn(qml.execute(tapes, qml.device("default.qubit", wires=1), None))
    tensor([0.98006658, 0.82533561, 0.54030231], requires_grad=True)
    """
    if tape.batch_size is None:
        output_tapes = [tape]

        def null_postprocessing(results):
            """A postprocesing function returned by a transform that only converts the batch of results
            into a result for a single ``QuantumTape``.
            """
            return results[0]
        processing_fn = null_postprocessing
    else:
        num_tapes = tape.batch_size
        new_ops = _split_operations(tape.operations, num_tapes)
        output_tapes = []
        for ops in new_ops:
            new_tape = qml.tape.QuantumScript(ops, tape.measurements, shots=tape.shots, trainable_params=tape.trainable_params)
            output_tapes.append(new_tape)

        def processing_fn(results: qml.typing.ResultBatch) -> qml.typing.Result:
            if len(tape.measurements) > 1:
                processed_results = [qml.math.squeeze(qml.math.stack([results[b][i] for b in range(tape.batch_size)])) for i in range(len(tape.measurements))]
                return tuple(processed_results)
            return qml.math.squeeze(qml.math.stack(results))
    return (output_tapes, processing_fn)