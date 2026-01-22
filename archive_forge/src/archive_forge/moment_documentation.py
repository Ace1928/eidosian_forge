import itertools
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, ops, qis, _compat
from cirq._import import LazyLoader
from cirq.ops import raw_types, op_tree
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
Determines whether Moment commutes with the Operation.

        Args:
            other: An Operation object. Other types are not implemented yet.
                In case a different type is specified, NotImplemented is
                returned.
            atol: Absolute error tolerance. If all entries in v1@v2 - v2@v1
                have a magnitude less than this tolerance, v1 and v2 can be
                reported as commuting. Defaults to 1e-8.

        Returns:
            True: The Moment and Operation commute OR they don't have shared
            quibits.
            False: The two values do not commute.
            NotImplemented: In case we don't know how to check this, e.g.
                the parameter type is not supported yet.
        