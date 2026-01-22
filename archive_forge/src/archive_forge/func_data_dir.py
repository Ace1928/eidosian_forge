import abc
import dataclasses
import os
from typing import Any, Dict, List, TYPE_CHECKING
import cirq
from cirq import _compat
from cirq.protocols import dataclass_json_dict
@property
def data_dir(self) -> str:
    """The data directory, namely '{base_data_dir}/{run_id}"""
    return self._data_dir