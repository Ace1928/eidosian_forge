import collections
from functools import wraps
from itertools import count
from inspect import getfullargspec as getArgsSpec
import attr
from ._core import Transitioner, Automaton
from ._introspection import preserveName
@preserveName(self.method)
@wraps(self.method)
def doInput(*args, **kwargs):
    self.method(oself, *args, **kwargs)
    previousState = transitioner._state
    outputs, outTracer = transitioner.transition(self)
    collector = self.collectors[previousState]
    values = []
    for output in outputs:
        if outTracer:
            outTracer(output._name())
        a, k = _filterArgs(args, kwargs, self.argSpec, output.argSpec)
        value = output(oself, *a, **k)
        values.append(value)
    return collector(values)