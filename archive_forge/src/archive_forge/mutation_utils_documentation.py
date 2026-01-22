import collections
from typing import cast, Dict, List, Optional, Sequence, Union, TYPE_CHECKING
from cirq import circuits, ops, transformers
from cirq.contrib.acquaintance.gates import SwapNetworkGate, AcquaintanceOpportunityGate
from cirq.contrib.acquaintance.devices import get_acquaintance_size
from cirq.contrib.acquaintance.permutation import PermutationGate
Decomposes permutation gates that provide acquaintance opportunities.