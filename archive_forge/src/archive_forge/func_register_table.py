from __future__ import annotations
import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Set, Tuple, TYPE_CHECKING, Union
from torch._inductor import config
from torch._inductor.utils import get_benchmark_name
@staticmethod
def register_table(name, column_names):
    table = MetricTable(name, column_names)
    REGISTERED_METRIC_TABLES[name] = table