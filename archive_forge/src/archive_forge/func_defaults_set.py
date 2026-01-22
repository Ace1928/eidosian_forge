import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def defaults_set(self) -> bool:
    """Whether the default angles are set.

        This only considers angles where characterize_{angle} is True. If all such angles have
        {angle}_default set to a value, this returns True. If none of the defaults are set,
        this returns False. If some defaults are set, we raise an exception.
        """
    defaults_set = [default is not None for _, default, _ in self._iter_angles()]
    if any(defaults_set):
        if all(defaults_set):
            return True
        problems = [symbol.name for _, default, symbol in self._iter_angles() if default is None]
        raise ValueError(f'Some angles are set, but values for {problems} are not.')
    return False