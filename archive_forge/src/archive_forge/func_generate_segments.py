import sys
from itertools import chain
from typing import TYPE_CHECKING, Iterable, Optional
from .constrain import Constrain
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment
from .style import StyleType
def generate_segments() -> Iterable[Segment]:
    if excess_space <= 0:
        for line in lines:
            yield from line
            yield new_line
    elif align == 'left':
        pad = Segment(' ' * excess_space, style) if self.pad else None
        for line in lines:
            yield from line
            if pad:
                yield pad
            yield new_line
    elif align == 'center':
        left = excess_space // 2
        pad = Segment(' ' * left, style)
        pad_right = Segment(' ' * (excess_space - left), style) if self.pad else None
        for line in lines:
            if left:
                yield pad
            yield from line
            if pad_right:
                yield pad_right
            yield new_line
    elif align == 'right':
        pad = Segment(' ' * excess_space, style)
        for line in lines:
            yield pad
            yield from line
            yield new_line