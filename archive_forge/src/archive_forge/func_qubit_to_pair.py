import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
@property
@lru_cache_typesafe
def qubit_to_pair(self) -> MutableMapping[cirq.Qid, Tuple[cirq.Qid, cirq.Qid]]:
    """Returns mapping from qubit to a qubit pair that it belongs to."""
    return collections.ChainMap(*({q: pair for q in pair} for pair in self.pairs))