from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
def get_as(self, loc: Literal['t', 'b', 'l', 'r'], units: Literal['pt', 'in', 'lines', 'fig']='pt') -> float:
    """
        Return key in given units
        """
    dpi = 72
    size: float = self.theme.getp((self.themeable_name, 'size'), 11)
    from_units = self.units
    to_units = units
    W: float
    H: float
    W, H = self.theme.getp('figure_size')
    L = W * dpi if loc in 'tb' else H * dpi
    functions: dict[str, Callable[[float], float]] = {'fig-in': lambda x: x * L / dpi, 'fig-lines': lambda x: x * L / size, 'fig-pt': lambda x: x * L, 'in-fig': lambda x: x * dpi / L, 'in-lines': lambda x: x * dpi / size, 'in-pt': lambda x: x * dpi, 'lines-fig': lambda x: x * size / L, 'lines-in': lambda x: x * size / dpi, 'lines-pt': lambda x: x * size, 'pt-fig': lambda x: x / L, 'pt-in': lambda x: x / dpi, 'pt-lines': lambda x: x / size}
    value: float = getattr(self, loc)
    if from_units != to_units:
        conversion = f'{self.units}-{units}'
        try:
            value = functions[conversion](value)
        except ZeroDivisionError:
            value = 0
    return value