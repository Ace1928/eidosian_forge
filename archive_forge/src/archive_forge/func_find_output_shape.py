import itertools
from collections import OrderedDict
import numpy as np
def find_output_shape(inputs, shapes, output):
    """Find the output shape for given inputs, shapes and output string, taking
    into account broadcasting.

    Examples
    --------
    >>> oe.parser.find_output_shape(["ab", "bc"], [(2, 3), (3, 4)], "ac")
    (2, 4)

    # Broadcasting is accounted for
    >>> oe.parser.find_output_shape(["a", "a"], [(4, ), (1, )], "a")
    (4,)
    """
    return tuple((max((shape[loc] for shape, loc in zip(shapes, [x.find(c) for x in inputs]) if loc >= 0)) for c in output))