from __future__ import annotations
import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Set, Tuple, TYPE_CHECKING, Union
from torch._inductor import config
from torch._inductor.utils import get_benchmark_name
def purge_old_log_files():
    """
    Purge the old log file at the beginning when the benchmark script runs.
    Should do it in the parent process rather than the child processes running
    each individual model.
    """
    for name, table in REGISTERED_METRIC_TABLES.items():
        if name in enabled_metric_tables():
            filename = table.output_filename()
            if os.path.exists(filename):
                os.unlink(filename)
            table.write_header()