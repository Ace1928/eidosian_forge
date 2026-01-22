import sys
import re
import functools
import os
import contextlib
import warnings
import inspect
import pathlib
from typing import Any, Callable
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.exceptions import ignore_warnings # noqa:F401
@contextlib.contextmanager
def warns_deprecated_sympy():
    """
    Shorthand for ``warns(SymPyDeprecationWarning)``

    This is the recommended way to test that ``SymPyDeprecationWarning`` is
    emitted for deprecated features in SymPy. To test for other warnings use
    ``warns``. To suppress warnings without asserting that they are emitted
    use ``ignore_warnings``.

    .. note::

       ``warns_deprecated_sympy()`` is only intended for internal use in the
       SymPy test suite to test that a deprecation warning triggers properly.
       All other code in the SymPy codebase, including documentation examples,
       should not use deprecated behavior.

       If you are a user of SymPy and you want to disable
       SymPyDeprecationWarnings, use ``warnings`` filters (see
       :ref:`silencing-sympy-deprecation-warnings`).

    >>> from sympy.testing.pytest import warns_deprecated_sympy
    >>> from sympy.utilities.exceptions import sympy_deprecation_warning
    >>> with warns_deprecated_sympy():
    ...     sympy_deprecation_warning("Don't use",
    ...        deprecated_since_version="1.0",
    ...        active_deprecations_target="active-deprecations")

    >>> with warns_deprecated_sympy():
    ...     pass
    Traceback (most recent call last):
    ...
    Failed: DID NOT WARN. No warnings of type     SymPyDeprecationWarning was emitted. The list of emitted warnings is: [].

    .. note::

       Sometimes the stacklevel test will fail because the same warning is
       emitted multiple times. In this case, you can use
       :func:`sympy.utilities.exceptions.ignore_warnings` in the code to
       prevent the ``SymPyDeprecationWarning`` from being emitted again
       recursively. In rare cases it is impossible to have a consistent
       ``stacklevel`` for deprecation warnings because different ways of
       calling a function will produce different call stacks.. In those cases,
       use ``warns(SymPyDeprecationWarning)`` instead.

    See Also
    ========
    sympy.utilities.exceptions.SymPyDeprecationWarning
    sympy.utilities.exceptions.sympy_deprecation_warning
    sympy.utilities.decorator.deprecated

    """
    with warns(SymPyDeprecationWarning):
        yield