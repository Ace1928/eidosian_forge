import json
import numbers
from typing import Any
from pennylane.wires import Wires
def wires_to_json(wires: Wires) -> str:
    """Converts ``wires`` to a JSON list, with wire labels in
    order of their index.

    Returns:
        JSON list of wires

    Raises:
        UnserializableWireError: if any of the wires are not JSON-serializable.
    """
    jsonable_wires = []
    for w in wires:
        if type(w) in _JSON_TYPES:
            jsonable_wires.append(w)
        elif isinstance(w, numbers.Integral):
            w_converted = int(w)
            if hash(w_converted) != hash(w):
                raise UnserializableWireError(w)
            jsonable_wires.append(w_converted)
        else:
            raise UnserializableWireError(w)
    return json.dumps(jsonable_wires)