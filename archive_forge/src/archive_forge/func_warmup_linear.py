import contextlib
import dataclasses
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Generic, Iterable, List, Optional, TypeVar, Union
import catalogue
import confection
@my_registry.schedules('warmup_linear.v1')
def warmup_linear(initial_rate: float, warmup_steps: int, total_steps: int) -> Iterable[float]:
    """Generate a series, starting from an initial rate, and then with a warmup
    period, and then a linear decline. Used for learning rates.
    """
    step = 0
    while True:
        if step < warmup_steps:
            factor = step / max(1, warmup_steps)
        else:
            factor = max(0.0, (total_steps - step) / max(1.0, total_steps - warmup_steps))
        yield (factor * initial_rate)
        step += 1