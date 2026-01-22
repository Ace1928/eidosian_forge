import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def insert_equiv(self, *objs):
    """Overload EquivSet.insert_equiv to handle Numba IR variables and
        constants. Input objs are either variable or constant, and at least
        one of them must be variable.
        """
    assert len(objs) > 1
    obj_names = [self._get_names(x) for x in objs]
    obj_names = [x for x in obj_names if x != ()]
    if len(obj_names) <= 1:
        return
    names = sum([list(x) for x in obj_names], [])
    ndims = [len(x) for x in obj_names]
    ndim = ndims[0]
    assert all((ndim == x for x in ndims)), 'Dimension mismatch for {}'.format(objs)
    varlist = []
    constlist = []
    for obj in objs:
        if not isinstance(obj, tuple):
            obj = (obj,)
        for var in obj:
            if isinstance(var, ir.Var) and (not var.name in varlist):
                if var.name in self.defs:
                    varlist.insert(0, var)
                else:
                    varlist.append(var)
            if isinstance(var, ir.Const) and (not var.value in constlist):
                constlist.append(var.value)
    for obj in varlist:
        name = obj.name
        if name in names and (not name in self.obj_to_ind):
            self.ind_to_obj[self.next_ind] = [name]
            self.obj_to_ind[name] = self.next_ind
            self.ind_to_var[self.next_ind] = [obj]
            self.next_ind += 1
    for const in constlist:
        if const in names and (not const in self.obj_to_ind):
            self.ind_to_obj[self.next_ind] = [const]
            self.obj_to_ind[const] = self.next_ind
            self.ind_to_const[self.next_ind] = const
            self.next_ind += 1
    some_change = False
    for i in range(ndim):
        names = [obj_name[i] for obj_name in obj_names]
        ie_res = super(ShapeEquivSet, self).insert_equiv(*names)
        some_change = some_change or ie_res
    return some_change