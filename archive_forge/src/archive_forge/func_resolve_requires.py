import timeit
from abc import abstractmethod, ABCMeta
from collections import namedtuple, OrderedDict
import inspect
from pprint import pformat
from numba.core.compiler_lock import global_compiler_lock
from numba.core import errors, config, transforms, utils
from numba.core.tracing import event
from numba.core.postproc import PostProcessor
from numba.core.ir_utils import enforce_no_dels, legalize_single_scope
import numba.core.event as ev
def resolve_requires(key, rmap):

    def walk(lkey, rmap):
        dep_set = rmap[lkey] if lkey in rmap else set()
        if dep_set:
            for x in dep_set:
                dep_set |= walk(x, rmap)
            return dep_set
        else:
            return set()
    ret = set()
    for k in key:
        ret |= walk(k, rmap)
    return ret