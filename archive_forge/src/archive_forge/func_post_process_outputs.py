from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import operator
import warnings
from functools import reduce
import numpy as np
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder, parse_identity
from numba.core import types, sigutils
from numba.core.typing import signature
from numba.np.ufunc.sigparse import parse_signature
def post_process_outputs(self, outputs):
    """
        Moves the given output(s) to the host if necessary.

        Returns a single value (e.g. an array) if there was one output, or a
        tuple of arrays if there were multiple. Although this feels a little
        jarring, it is consistent with the behavior of GUFuncs in general.
        """
    if self._copy_result_to_host:
        outputs = [self.to_host(output, self_output) for output, self_output in zip(outputs, self.outputs)]
    elif self.outputs[0] is not None:
        outputs = self.outputs
    if len(outputs) == 1:
        return outputs[0]
    else:
        return tuple(outputs)