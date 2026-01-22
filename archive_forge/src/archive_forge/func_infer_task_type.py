from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def infer_task_type(df: pd.DataFrame) -> str:
    """
    Infer the likely fine-tuning task type from the data
    """
    CLASSIFICATION_THRESHOLD = 3
    if sum(df.prompt.str.len()) == 0:
        return 'open-ended generation'
    if len(df.completion.unique()) < len(df) / CLASSIFICATION_THRESHOLD:
        return 'classification'
    return 'conditional generation'