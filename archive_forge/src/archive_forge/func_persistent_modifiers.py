import abc
from typing import (
from cirq import ops, value, devices
def persistent_modifiers(self) -> Dict[str, Callable[['Cell'], 'Cell']]:
    """Overridable modifications to apply to the rest of the circuit.

        Persistent modifiers apply to all cells in the same column and also to
        all cells in future columns (until a column overrides the modifier with
        another one using the same key).

        Returns:
            A dictionary of keyed modifications. Each modifier lasts until a
            later cell specifies a new modifier with the same key.
        """
    return {}