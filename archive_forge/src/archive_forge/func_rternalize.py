import abc
import atexit
import contextlib
import contextvars
import csv
import enum
import functools
import inspect
import os
import math
import platform
import signal
import subprocess
import textwrap
import threading
import typing
import warnings
from typing import Union
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface_lib._rinterface_capi as _rinterface
import rpy2.rinterface_lib.embedded as embedded
import rpy2.rinterface_lib.conversion as conversion
from rpy2.rinterface_lib.conversion import _cdata_res_to_rinterface
import rpy2.rinterface_lib.memorymanagement as memorymanagement
from rpy2.rinterface_lib import na_values
from rpy2.rinterface_lib.sexp import NULL
from rpy2.rinterface_lib.sexp import NULLType
import rpy2.rinterface_lib.bufferprotocol as bufferprotocol
from rpy2.rinterface_lib import sexp
from rpy2.rinterface_lib.sexp import CharSexp  # noqa: F401
from rpy2.rinterface_lib.sexp import RTYPES
from rpy2.rinterface_lib.sexp import SexpVector
from rpy2.rinterface_lib.sexp import StrSexpVector
from rpy2.rinterface_lib.sexp import Sexp
from rpy2.rinterface_lib.sexp import SexpEnvironment
from rpy2.rinterface_lib.sexp import unserialize  # noqa: F401
from rpy2.rinterface_lib.sexp import emptyenv
from rpy2.rinterface_lib.sexp import baseenv
from rpy2.rinterface_lib.sexp import globalenv
def rternalize(function: typing.Optional[typing.Callable]=None, *, signature: bool=False) -> typing.Union[SexpClosure, functools.partial]:
    """ Make a Python function callable from R.

    Takes an arbitrary Python function and wrap it in such a way that
    it can be called from the R side.

    This factory can be used as a decorator, and has an optional
    named argument "signature" that can be True or False (default is
    False). When True, the R function wrapping the Python one will
    have a matching signature or a close one, as detailed below.

    The Python ellipsis arguments`*args` and `**kwargs` are
    translated to the R ellipsis `...`.

    For example:

    .. code-block:: python
       @rternalize(signature=True)
       def foo(x, *args, y=2):
           pass

    will be visible in R with the signature:

    .. code-block:: r
       function (x, ..., y)

    The only limitation is that whenever `*args` and `**kwargs` are
    both present in the Python declaration they must be consecutive.
    For example:

    .. code-block:: python
       def foo(x, *args, y=2, **kwargs):
           pass

    is a valid Python declaration. However, it cannot be "rternalized"
    because the R ellipsis can only appear at most once in the signature
    of a given function. Trying to apply the decorator `rternalize` would
    raise an exception.

    The following Python function can be "rternalized":

    .. code-block:: python
       def foo(x, *args, **kwargs):
           pass

    It is visible to R with the signature

    .. code-block:: r
       function (x, ...)

    Python function definitions can allow the optional naming of required
    arguments. The mapping of signatures between Python and R is then
    quasi-indentical since R does it for unnamed arguments. The check
    that all arguments are present is still performed on the Python side.

    Example:

    .. code-block:: python
       @rternalize(signature=True)
       def foo(x, *, y, z):
           print(f'x: {x[0]}, y: {y[0]}, z: {z[0]}')
           return ri.NULL

    >>> _ = foo(1, 2, 3)
    x: 1, y: 2, z: 3
    ValueErro: None
    >>> _ = foo(1)
    TypeError: rternalized foo() missing 2 required keyword-only arguments: 'y' and 'z'
    >>> _ = foo(1, z=2, y=3)
    x: 1, y: 3, z: 2
    >>> _ = foo(1, z=2, y=3)
    x: 1, y: 3, z: 2

    Note that whenever the Python function has an ellipsis (either `*args`
    or `**kwargs`) earlier parameters in the signature that are
    positional-or-keyword are considered to be positional arguments in a
    function call.

    :param function: A Python callable object. This is a positional
    argument with a default value `None` to allow the decorator function
    without parentheses when optional argument is not wanted.

    :return: A wrapped R object that can be use like any other rpy2
    object.
    """
    if not embedded.isinitialized():
        raise embedded.RNotReadyError('The embedded R is not yet initialized.')
    if function is None:
        return functools.partial(rternalize, signature=signature)
    assert callable(function)
    rpy_fun = SexpExtPtr.from_pyobject(function)
    if not signature:
        template = parse('\n        function(...) { .External(".Python", foo, ...);\n        }\n        ')
        template[0][2][1][2] = rpy_fun
    else:
        has_ellipsis = None
        params_r_sig = []
        for p_i, (name, param) in enumerate(inspect.signature(function).parameters.items()):
            if param.kind is inspect.Parameter.VAR_POSITIONAL or param.kind is inspect.Parameter.VAR_KEYWORD:
                if has_ellipsis:
                    if has_ellipsis != p_i - 1:
                        raise ValueError('R functions can only have one ellipsis. As consequence your Python function must have *args and **kwargs that are consecutive in function signature.')
                else:
                    has_ellipsis = p_i
                    params_r_sig.append('...')
            else:
                params_r_sig.append(name)
        r_func_args = ', '.join(params_r_sig)
        r_src = f'\n        function({r_func_args}) {{\n            py_func <- RPY2_FUN_PLACEHOLDER\n            lst_args <- base::as.list(base::match.call()[-1])\n            RPY2_ARGUMENTS <- base::c(\n                base::list(\n                    ".Python",\n                    py_func\n                ),\n                lst_args\n            )\n            res <- base::do.call(\n               base::.External,\n               RPY2_ARGUMENTS\n            );\n\n            res\n        }}\n        '
        template = parse(r_src)
        function_definition = _find_first(template, of_type=LangSexpVector)
        function_body = _find_first(function_definition, of_type=LangSexpVector)
        rpy_fun_node = function_body[1]
        assert str(rpy_fun_node[2]) == 'RPY2_FUN_PLACEHOLDER'
        rpy_fun_node[2] = rpy_fun
    res = baseenv['eval'](template)
    res.__nested_sexp__ = rpy_fun.__sexp__
    return res