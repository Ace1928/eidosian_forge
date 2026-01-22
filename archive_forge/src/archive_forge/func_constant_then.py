from typing import Iterable
import numpy
from .config import registry
@registry.schedules('constant_then.v1')
def constant_then(rate: float, steps: int, schedule: Iterable[float]) -> Iterable[float]:
    """Yield a constant rate for N steps, before starting a schedule."""
    for i in range(steps):
        yield rate
    for value in schedule:
        yield value