import re, copy
from math import acos, ceil, copysign, cos, degrees, fabs, hypot, radians, sin, sqrt
from .shapes import Group, mmult, rotate, translate, transformPoint, Path, FILL_EVEN_ODD, _CLOSEPATH, UserNode
def normalise_svg_path(attr):
    """Normalise SVG path.

    This basically introduces operator codes for multi-argument
    parameters. Also, it fixes sequences of consecutive M or m
    operators to MLLL... and mlll... operators. It adds an empty
    list as argument for Z and z only in order to make the resul-
    ting list easier to iterate over.

    E.g. "M 10 20, M 20 20, L 30 40, 40 40, Z"
      -> ['M', [10, 20], 'L', [20, 20], 'L', [30, 40], 'L', [40, 40], 'Z', []]
    """
    ops = {'A': 7, 'a': 7, 'Q': 4, 'q': 4, 'T': 2, 't': 2, 'S': 4, 's': 4, 'M': 2, 'L': 2, 'm': 2, 'l': 2, 'H': 1, 'V': 1, 'h': 1, 'v': 1, 'C': 6, 'c': 6, 'Z': 0, 'z': 0}
    op_keys = ops.keys()
    result = []
    groups = re.split('([achlmqstvz])', attr.strip(), flags=re.I)
    op = None
    for item in groups:
        if item.strip() == '':
            continue
        if item in op_keys:
            if item == 'M' and item == op:
                op = 'L'
            elif item == 'm' and item == op:
                op = 'l'
            else:
                op = item
            if ops[op] == 0:
                result.extend([op, []])
        else:
            if op.lower() == 'a':
                result.extend(split_arc_values(op, item))
            else:
                result.extend(split_floats(op, ops[op], item))
            op = result[-2]
    return result