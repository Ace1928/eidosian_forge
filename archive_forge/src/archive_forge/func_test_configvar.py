import json
from typing import Callable, cast
from adagio.exceptions import SkippedError
from adagio.instances import (_ConfigVar, _Dependency, _DependencyDict, _Input,
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec
from pytest import raises
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.hash import to_uuid
def test_configvar():
    t = MockTaskForVar()
    s = ConfigSpec('a', dict, True, True, None)
    c = _ConfigVar(t, s)
    raises(AssertionError, lambda: c.get())
    p = ParamDict()
    s = ConfigSpec('a', dict, True, False, p)
    c = _ConfigVar(t, s)
    assert p is c.get()
    c.set(None)
    assert c.get() is None
    p = ParamDict()
    s = ConfigSpec('a', ParamDict, False, False, p)
    c = _ConfigVar(t, s)
    assert p is c.get()
    raises(AssertionError, lambda: c.set(None))
    assert p is c.get()
    p2 = ParamDict()
    s2 = ConfigSpec('x', dict, False, False, p2)
    c2 = _ConfigVar(t, s2)
    assert p2 is c2.get()
    c2.set_dependency(c)
    assert p is c2.get()
    p3 = ParamDict()
    c.set(p3)
    assert p3 is c.get()
    assert p3 is c2.get()