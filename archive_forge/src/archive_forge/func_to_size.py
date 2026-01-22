import datetime
import importlib
import inspect
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import six  # type: ignore
from triad.utils.assertion import assert_or_throw
def to_size(exp: Any) -> int:
    """Convert input value or expression to size
    For expression string, it must be in the format of
    `<value>` or `<value><unit>`. Value must be 0 or positive,
    default unit is byte if not provided. Unit can be `b`, `byte`,
    `k`, `kb`, `m`, `mb`, `g`, `gb`, `t`, `tb`.

    Args:
        exp (Any): expression string or numerical value

    Raises:
        ValueError: for invalid expression
        ValueError: for negative values

    Returns:
        int: size in byte
    """
    n, u = _parse_value_and_unit(exp)
    assert n >= 0.0, "Size can't be negative"
    if u in ['', 'b', 'byte', 'bytes']:
        return int(n)
    if u in ['k', 'kb']:
        return int(n * 1024)
    if u in ['m', 'mb']:
        return int(n * 1024 * 1024)
    if u in ['g', 'gb']:
        return int(n * 1024 * 1024 * 1024)
    if u in ['t', 'tb']:
        return int(n * 1024 * 1024 * 1024 * 1024)
    raise ValueError(f'Invalid size expression {exp}')