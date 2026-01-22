from typing import TYPE_CHECKING
import abc
from cirq_google.line.placement.sequence import GridQubitLineTuple
Runs line sequence search.

        Args:
            device: Chip description.
            length: Required line length.

        Returns:
            Linear sequences found on the chip.
        