from collections import abc
from typing import Any, Dict, Generic, Iterator, List, Mapping, Optional, Sequence, TYPE_CHECKING
import numpy as np
from cirq import ops, protocols, value
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase
@property
def split_untangled_states(self) -> bool:
    return self._split_untangled_states