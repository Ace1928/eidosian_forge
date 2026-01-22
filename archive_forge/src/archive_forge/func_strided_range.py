import pyglet.gl as pgl
from sympy.core import S
def strided_range(r_min, r_max, stride, max_steps=50):
    o_min, o_max = (r_min, r_max)
    if abs(r_min - r_max) < 0.001:
        return []
    try:
        range(int(r_min - r_max))
    except (TypeError, OverflowError):
        return []
    if r_min > r_max:
        raise ValueError('r_min cannot be greater than r_max')
    r_min_s = r_min % stride
    r_max_s = stride - r_max % stride
    if abs(r_max_s - stride) < 0.001:
        r_max_s = 0.0
    r_min -= r_min_s
    r_max += r_max_s
    r_steps = int((r_max - r_min) / stride)
    if max_steps and r_steps > max_steps:
        return strided_range(o_min, o_max, stride * 2)
    return [r_min] + [r_min + e * stride for e in range(1, r_steps + 1)] + [r_max]