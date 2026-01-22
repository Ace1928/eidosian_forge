import dataclasses
import inspect
from types import FunctionType
from typing import Any
from typing import final
from typing import Generic
from typing import Type
from typing import TypeVar
import warnings
def warn_explicit_for(method: FunctionType, message: PytestWarning) -> None:
    """
    Issue the warning :param:`message` for the definition of the given :param:`method`

    this helps to log warnings for functions defined prior to finding an issue with them
    (like hook wrappers being marked in a legacy mechanism)
    """
    lineno = method.__code__.co_firstlineno
    filename = inspect.getfile(method)
    module = method.__module__
    mod_globals = method.__globals__
    try:
        warnings.warn_explicit(message, type(message), filename=filename, module=module, registry=mod_globals.setdefault('__warningregistry__', {}), lineno=lineno)
    except Warning as w:
        raise type(w)(f'{w}\n at {filename}:{lineno}') from None