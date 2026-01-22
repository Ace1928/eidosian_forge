import copy
import copyreg as copy_reg
import inspect
import pickle
import types
from io import StringIO as _cStringIO
from typing import Dict
from twisted.python import log, reflect
from twisted.python.compat import _PYPY
def unpickleMethod(im_name, im_self, im_class):
    """
    Support function for copy_reg to unpickle method refs.

    @param im_name: The name of the method.
    @type im_name: native L{str}

    @param im_self: The instance that the method was present on.
    @type im_self: L{object}

    @param im_class: The class where the method was declared.
    @type im_class: L{type} or L{None}
    """
    if im_self is None:
        return getattr(im_class, im_name)
    try:
        methodFunction = _methodFunction(im_class, im_name)
    except AttributeError:
        log.msg('Method', im_name, 'not on class', im_class)
        assert im_self is not None, 'No recourse: no instance to guess from.'
        if im_self.__class__ is im_class:
            raise
        return unpickleMethod(im_name, im_self, im_self.__class__)
    else:
        maybeClass = ()
        bound = types.MethodType(methodFunction, im_self, *maybeClass)
        return bound