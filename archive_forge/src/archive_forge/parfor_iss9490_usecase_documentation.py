import numba
import numpy as np

This is a testcase for https://github.com/numba/numba/issues/9490.
The bug is very sensitive to the control-flow and variable uses.
It is impossible to shrink the reproducer in any meaningful way.

The test is also sensitive to PYTHONHASHSEED.
PYTHONHASHSEED=1 will trigger the bug.

Example of traceback:

  File "/numba/parfors/parfor.py", line 2070, in _arrayexpr_to_parfor
    index_vars, loopnests = _mk_parfor_loops(pass_states.typemap, size_vars,
                                             scope, loc)
  File "/numba/parfors/parfor.py", line 1981, in _mk_parfor_loops
    for size_var in size_vars:
TypeError: Failed in nopython mode pipeline (step: convert to parfors)
'NoneType' object is not iterable
