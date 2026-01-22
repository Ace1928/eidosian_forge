from collections import abc, defaultdict
import datetime
from itertools import cycle
from typing import Any, cast, Dict, Iterator, List, Optional, Tuple, Union, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
import google.protobuf.json_format as json_format
import cirq
from cirq_google.api import v2
@staticmethod
def key_to_qubit(target: METRIC_KEY) -> cirq.GridQubit:
    """Returns a single qubit from a metric key.

        Raises:
           ValueError: If the metric key is a tuple of strings.
        """
    if target and isinstance(target, tuple) and isinstance(target[0], cirq.GridQubit):
        return target[0]
    raise ValueError(f'The metric target {target} was not a tuple of qubits')