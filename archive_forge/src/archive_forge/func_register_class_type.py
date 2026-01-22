import inspect
import operator
import types as pytypes
import typing as pt
from collections import OrderedDict
from collections.abc import Sequence
from llvmlite import ir as llvmir
from numba import njit
from numba.core import cgutils, errors, imputils, types, utils
from numba.core.datamodel import default_manager, models
from numba.core.registry import cpu_target
from numba.core.typing import templates
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.serialize import disable_pickling
from numba.experimental.jitclass import _box
def register_class_type(cls, spec, class_ctor, builder):
    """
    Internal function to create a jitclass.

    Args
    ----
    cls: the original class object (used as the prototype)
    spec: the structural specification contains the field types.
    class_ctor: the numba type to represent the jitclass
    builder: the internal jitclass builder
    """
    if spec is None:
        spec = OrderedDict()
    elif isinstance(spec, Sequence):
        spec = OrderedDict(spec)
    for attr, py_type in pt.get_type_hints(cls).items():
        if attr not in spec:
            spec[attr] = as_numba_type(py_type)
    _validate_spec(spec)
    spec = _fix_up_private_attr(cls.__name__, spec)
    clsdct = {}
    for basecls in reversed(inspect.getmro(cls)):
        clsdct.update(basecls.__dict__)
    methods, props, static_methods, others = ({}, {}, {}, {})
    for k, v in clsdct.items():
        if isinstance(v, pytypes.FunctionType):
            methods[k] = v
        elif isinstance(v, property):
            props[k] = v
        elif isinstance(v, staticmethod):
            static_methods[k] = v
        else:
            others[k] = v
    shadowed = (set(methods) | set(props) | set(static_methods)) & set(spec)
    if shadowed:
        raise NameError('name shadowing: {0}'.format(', '.join(shadowed)))
    docstring = others.pop('__doc__', '')
    _drop_ignored_attrs(others)
    if others:
        msg = 'class members are not yet supported: {0}'
        members = ', '.join(others.keys())
        raise TypeError(msg.format(members))
    for k, v in props.items():
        if v.fdel is not None:
            raise TypeError('deleter is not supported: {0}'.format(k))
    jit_methods = {k: njit(v) for k, v in methods.items()}
    jit_props = {}
    for k, v in props.items():
        dct = {}
        if v.fget:
            dct['get'] = njit(v.fget)
        if v.fset:
            dct['set'] = njit(v.fset)
        jit_props[k] = dct
    jit_static_methods = {k: njit(v.__func__) for k, v in static_methods.items()}
    class_type = class_ctor(cls, ConstructorTemplate, spec, jit_methods, jit_props, jit_static_methods)
    jit_class_dct = dict(class_type=class_type, __doc__=docstring)
    jit_class_dct.update(jit_static_methods)
    cls = JitClassType(cls.__name__, (cls,), jit_class_dct)
    typingctx = cpu_target.typing_context
    typingctx.insert_global(cls, class_type)
    targetctx = cpu_target.target_context
    builder(class_type, typingctx, targetctx).register()
    as_numba_type.register(cls, class_type.instance_type)
    return cls