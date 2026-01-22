from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def set_on_model(self, on_model):
    """Register a callback that is invoked with every incremental improvement to
        objective values. The callback takes a model as argument.
        The life-time of the model is limited to the callback so the
        model has to be (deep) copied if it is to be used after the callback
        """
    id = len(_on_models) + 41
    mdl = Model(self.ctx)
    _on_models[id] = (on_model, mdl)
    self._on_models_id = id
    Z3_optimize_register_model_eh(self.ctx.ref(), self.optimize, mdl.model, ctypes.c_void_p(id), _on_model_eh)