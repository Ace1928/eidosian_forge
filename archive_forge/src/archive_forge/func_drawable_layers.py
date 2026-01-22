from pennylane.ops import Conditional
from pennylane.measurements import MidMeasureMP, MeasurementProcess
from .utils import default_wire_map
def drawable_layers(operations, wire_map=None, bit_map=None):
    """Determine non-overlapping yet dense placement of operations into layers for drawing.

    Args:
        operations (Iterable[~.Operator]): A list of operations.

    Keyword Args:
        wire_map (dict): A map from wire label to non-negative integers. Defaults to None.
        bit_map (dict): A map containing mid-circuit measurements used for classical conditions
            or collecting statistics as keys. Defaults to None.

    Returns:
        (list[set[~.Operator]], list[set[~.MeasurementProcess]]) : Each index is a set of operations
        for the corresponding layer in both lists. The first list corresponds to the operation layers,
        and the second corresponds to the measurement layers.

    **Details**

    The function recursively pushes operations as far to the left (lowest layer) possible
    *without* altering order.

    From the start, the function cares about the locations the operation altered
    during a drawing, not just the wires the operation acts on. An "occupied" wire
    refers to a wire that will be altered in the drawing of an operation.
    Assuming wire ``1`` is between ``0`` and ``2`` in the ordering, ``qml.CNOT(wires=(0,2))``
    will also "occupy" wire ``1``.  In this scenario, an operation on wire ``1``, like
    ``qml.X(1)``, will not be pushed to the left
    of the ``qml.CNOT(wires=(0,2))`` gate, but be blocked by the occupied wire. This preserves
    ordering and makes placement more intuitive.

    The ``wire_order`` keyword argument used by user facing functions like :func:`~.draw` maps position
    to wire label.   The ``wire_map`` keyword argument used here maps label to position.
    The utility function :func:`~.circuit_drawer.utils.convert_wire_order` can perform this
    transformation.

    """
    wire_map = wire_map or default_wire_map(operations)
    bit_map = bit_map or {}
    max_layer = 0
    occupied_wires_per_layer = [set()]
    ops_in_layer = [[]]
    used_cwires_per_layer = [set()]
    for op in operations:
        if isinstance(op, MidMeasureMP):
            if len(op.wires) > 1:
                raise ValueError('Cannot draw mid-circuit measurements with more than one wire.')
        if isinstance(op, MeasurementProcess) and op.mv is not None:
            op_occupied_wires = set()
            mapped_cwires = [bit_map[m.measurements[0]] for m in op.mv] if isinstance(op.mv, list) else [bit_map[m] for m in op.mv.measurements]
            op_occupied_cwires = set(range(min(mapped_cwires), max(mapped_cwires) + 1))
            op_layer = _recursive_find_mcm_stats_layer(max_layer, op_occupied_cwires, used_cwires_per_layer, bit_map)
        else:
            op_occupied_wires = _get_op_occupied_wires(op, wire_map, bit_map)
            op_layer = _recursive_find_layer(max_layer, op_occupied_wires, occupied_wires_per_layer)
            op_occupied_cwires = set()
        if op_layer > max_layer:
            max_layer += 1
            occupied_wires_per_layer.append(set())
            ops_in_layer.append([])
            used_cwires_per_layer.append(set())
        ops_in_layer[op_layer].append(op)
        occupied_wires_per_layer[op_layer].update(op_occupied_wires)
        used_cwires_per_layer[op_layer].update(op_occupied_cwires)
    return list(filter(None, ops_in_layer[:-1])) + ops_in_layer[-1:]