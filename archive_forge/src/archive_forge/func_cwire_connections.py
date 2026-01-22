from pennylane.ops import Controlled, Conditional
from pennylane.measurements import MeasurementProcess, MidMeasureMP, MeasurementValue
def cwire_connections(layers, bit_map):
    """Extract the information required for classical control wires.

    Args:
        layers (List[List[.Operator, .MeasurementProcess]]): the operations and measurements sorted
            into layers via ``drawable_layers``. Measurement layers may be appended to operation layers.
        bit_map (Dict): Dictionary containing mid-circuit measurements that are used for
            classical conditions or measurement statistics as keys.

    Returns:
        list, list: list of list of accessed layers for each classical wire, and largest wire
        corresponding to the accessed layers in the list above.

    >>> with qml.queuing.AnnotatedQueue() as q:
    ...     m0 = qml.measure(0)
    ...     m1 = qml.measure(1)
    ...     qml.cond(m0 & m1, qml.Y)(0)
    ...     qml.cond(m0, qml.S)(3)
    >>> tape = qml.tape.QuantumScript.from_queue(q)
    >>> layers = drawable_layers(tape)
    >>> bit_map, cwire_layers, cwire_wires = cwire_connections(layers)
    >>> bit_map
    {measure(wires=[0]): 0, measure(wires=[1]): 1}
    >>> cwire_layers
    [[0, 2, 3], [1, 2]]
    >>> cwire_wires
    [[0, 0, 3], [1, 0]]

    From this information, we can see that the first classical wire is active in layers
    0, 2, and 3 while the second classical wire is active in layers 1 and 2.  The first "active"
    layer will always be the one with the mid circuit measurement.

    """
    if len(bit_map) == 0:
        return ([], [])
    connected_layers = [[] for _ in bit_map]
    connected_wires = [[] for _ in bit_map]
    for layer_idx, layer in enumerate(layers):
        for op in layer:
            if isinstance(op, MidMeasureMP) and op in bit_map:
                connected_layers[bit_map[op]].append(layer_idx)
                connected_wires[bit_map[op]].append(op.wires[0])
            elif isinstance(op, Conditional):
                for m in op.meas_val.measurements:
                    cwire = bit_map[m]
                    connected_layers[cwire].append(layer_idx)
                    connected_wires[cwire].append(max(op.wires))
            elif isinstance(op, MeasurementProcess) and op.mv is not None:
                if isinstance(op.mv, MeasurementValue):
                    for m in op.mv.measurements:
                        cwire = bit_map[m]
                        connected_layers[cwire].append(layer_idx)
                else:
                    for m in op.mv:
                        cwire = bit_map[m.measurements[0]]
                        connected_layers[cwire].append(layer_idx)
    return (connected_layers, connected_wires)