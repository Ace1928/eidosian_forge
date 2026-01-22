from typing import Iterable
import numpy
from .config import registry
@registry.schedules('cyclic_triangular.v1')
def cyclic_triangular(min_lr: float, max_lr: float, period: int) -> Iterable[float]:
    it = 1
    while True:
        cycle = numpy.floor(1 + it / (2 * period))
        x = numpy.abs(it / period - 2 * cycle + 1)
        relative = max(0, 1 - x)
        yield (min_lr + (max_lr - min_lr) * relative)
        it += 1