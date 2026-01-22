import inspect
import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
def register_db_case(case: ExportCase) -> None:
    """
    Registers a user provided ExportCase into example bank.
    """
    if case.name in _EXAMPLE_CASES:
        if case.name not in _EXAMPLE_CONFLICT_CASES:
            _EXAMPLE_CONFLICT_CASES[case.name] = [_EXAMPLE_CASES[case.name]]
        _EXAMPLE_CONFLICT_CASES[case.name].append(case)
        return
    _EXAMPLE_CASES[case.name] = case