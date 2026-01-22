import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
def fix_collisions(panels: List[Panel]) -> List[Panel]:
    x_max = 24
    for i, p1 in enumerate(panels):
        for p2 in panels[i:]:
            if collides(p1, p2):
                x, y = shift(p1, p2)
                if p2.layout['x'] + p2.layout['w'] + x <= x_max:
                    p2.layout['x'] += x
                else:
                    p2.layout['y'] += y
                    p2.layout['x'] = 0
    return panels