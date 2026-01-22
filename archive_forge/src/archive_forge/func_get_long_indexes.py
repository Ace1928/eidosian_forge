from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def get_long_indexes(d: pd.DataFrame) -> Any:
    long_examples = d.apply(lambda x: len(x.prompt) + len(x.completion) > 10000, axis=1)
    return d.reset_index().index[long_examples].tolist()