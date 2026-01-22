import re, copy
from math import acos, ceil, copysign, cos, degrees, fabs, hypot, radians, sin, sqrt
from .shapes import Group, mmult, rotate, translate, transformPoint, Path, FILL_EVEN_ODD, _CLOSEPATH, UserNode
def split_floats(op, min_num, value):
    """Split `value`, a list of numbers as a string, to a list of float numbers.

    Also optionally insert a `l` or `L` operation depending on the operation
    and the length of values.
    Example: with op='m' and value='10,20 30,40,' the returned value will be
             ['m', [10.0, 20.0], 'l', [30.0, 40.0]]
    """
    floats = [float(seq) for seq in re.findall('(-?\\d*\\.?\\d*(?:[eE][+-]?\\d+)?)', value) if seq]
    res = []
    for i in range(0, len(floats), min_num):
        if i > 0 and op in {'m', 'M'}:
            op = 'l' if op == 'm' else 'L'
        res.extend([op, floats[i:i + min_num]])
    return res