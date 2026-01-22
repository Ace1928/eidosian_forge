from __future__ import annotations
from collections import (
import copy
from typing import (
import numpy as np
from pandas._libs.writers import convert_json_to_lines
import pandas as pd
from pandas import DataFrame
def convert_to_line_delimits(s: str) -> str:
    """
    Helper function that converts JSON lists to line delimited JSON.
    """
    if not s[0] == '[' and s[-1] == ']':
        return s
    s = s[1:-1]
    return convert_json_to_lines(s)