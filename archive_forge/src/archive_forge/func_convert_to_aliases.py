from __future__ import annotations
import dataclasses
from typing import Callable, Literal
from ._internal import _internal_dataclass
def convert_to_aliases(self) -> list[list[str | int]]:
    """Converts arguments to a list of lists containing string or integer aliases.

        Returns:
            The list of aliases.
        """
    aliases: list[list[str | int]] = []
    for c in self.choices:
        if isinstance(c, AliasPath):
            aliases.append(c.convert_to_aliases())
        else:
            aliases.append([c])
    return aliases