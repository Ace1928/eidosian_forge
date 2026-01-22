from typing import Any, Dict, FrozenSet, Iterable, Mapping, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import linalg, protocols, value
from cirq._compat import proper_repr
from cirq.ops import raw_types
@staticmethod
def from_mixture(mixture: 'protocols.SupportsMixture', key: Union[str, 'cirq.MeasurementKey', None]=None):
    """Creates a copy of a mixture with the given measurement key."""
    return MixedUnitaryChannel(mixture=list(protocols.mixture(mixture)), key=key)