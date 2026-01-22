from __future__ import annotations
import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Set, Tuple, TYPE_CHECKING, Union
from torch._inductor import config
from torch._inductor.utils import get_benchmark_name
def get_metric_table(name):
    assert name in REGISTERED_METRIC_TABLES, f'Metric table {name} is not defined'
    return REGISTERED_METRIC_TABLES[name]