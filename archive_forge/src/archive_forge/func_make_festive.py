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
def make_festive(pass_class):
    assert not self.is_registered(pass_class)
    assert not self._does_pass_name_alias(pass_class.name())
    pass_class.pass_id = self._id
    self._id += 1
    self._registry[pass_class] = pass_info(pass_class(), mutates_CFG, analysis_only)
    return pass_class