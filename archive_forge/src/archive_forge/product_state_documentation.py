import abc
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq._doc import document
The projector associated with this state expressed as a matrix.

        This is |s⟩⟨s| where |s⟩ is this state.
        