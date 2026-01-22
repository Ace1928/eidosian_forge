from collections import abc, defaultdict
import datetime
from itertools import cycle
from typing import Any, cast, Dict, Iterator, List, Optional, Tuple, Union, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
import google.protobuf.json_format as json_format
import cirq
from cirq_google.api import v2
def str_to_key(self, target: str) -> Union[cirq.GridQubit, str]:
    """Turns a string into a calibration key.

        Attempts to parse it as a GridQubit.  If this fails,
        returns the string itself.
        """
    try:
        return v2.grid_qubit_from_proto_id(target)
    except ValueError:
        return target