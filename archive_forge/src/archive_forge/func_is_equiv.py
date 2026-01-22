import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def is_equiv(self, *objs):
    """Overload EquivSet.is_equiv to handle Numba IR variables and
        constants.
        """
    assert len(objs) > 1
    obj_names = [self._get_names(x) for x in objs]
    obj_names = [x for x in obj_names if x != ()]
    if len(obj_names) <= 1:
        return False
    ndims = [len(names) for names in obj_names]
    ndim = ndims[0]
    if not all((ndim == x for x in ndims)):
        if config.DEBUG_ARRAY_OPT >= 1:
            print('is_equiv: Dimension mismatch for {}'.format(objs))
        return False
    for i in range(ndim):
        names = [obj_name[i] for obj_name in obj_names]
        if not super(ShapeEquivSet, self).is_equiv(*names):
            return False
    return True