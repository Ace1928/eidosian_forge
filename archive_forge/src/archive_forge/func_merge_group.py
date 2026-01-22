import collections
import contextlib
import dataclasses
import os
import shutil
import tempfile
import textwrap
import time
from typing import cast, Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple
import uuid
import torch
def merge_group(task_spec: TaskSpec, group: List['Measurement']) -> 'Measurement':
    times: List[float] = []
    for m in group:
        times.extend(m.times)
    return Measurement(number_per_run=1, raw_times=times, task_spec=task_spec, metadata=None)