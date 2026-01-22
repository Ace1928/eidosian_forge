import functools
import itertools
from collections.abc import Iterable, Sequence
import numpy as np
from pennylane.pytrees import register_pytree
@staticmethod
def unique_wires(list_of_wires):
    """Return the wires that are unique to any Wire object in the list.

        Args:
            list_of_wires (List[Wires]): list of Wires objects

        Returns:
            Wires: unique wires

        **Example**

        >>> wires1 = Wires([4, 0, 1])
        >>> wires2 = Wires([0, 2, 3])
        >>> wires3 = Wires([5, 3])
        >>> Wires.unique_wires([wires1, wires2, wires3])
        <Wires = [4, 1, 2, 5]>
        """
    for wires in list_of_wires:
        if not isinstance(wires, Wires):
            raise WireError(f'Expected a Wires object; got {wires} of type {type(wires)}.')
    label_sets = [wire.toset() for wire in list_of_wires]
    seen_ever = set()
    seen_once = set()
    for labels in label_sets:
        seen_once = (seen_once ^ labels) - (seen_ever - seen_once)
        seen_ever.update(labels)
    unique = []
    for wires in list_of_wires:
        for wire in wires.tolist():
            if wire in seen_once:
                unique.append(wire)
    return Wires(tuple(unique), _override=True)