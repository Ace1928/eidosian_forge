import logging
import warnings
from enum import Enum
from typing import Any, Callable, List, Optional, Union
from ray._private.pydantic_compat import (
from ray._private.utils import import_attr
from ray.serve._private.constants import (
from ray.util.annotations import Deprecated, PublicAPI
@property
def grpc_servicer_func_callable(self) -> List[Callable]:
    """Return a list of callable functions from the grpc_servicer_functions.

        If the function is not callable or not found, it will be ignored and a warning
        will be logged.
        """
    callables = []
    for func in self.grpc_servicer_functions:
        try:
            imported_func = import_attr(func)
            if callable(imported_func):
                callables.append(imported_func)
            else:
                message = f'{func} is not a callable function! Please make sure the function is imported correctly.'
                raise ValueError(message)
        except ModuleNotFoundError as e:
            message = f"{func} can't be imported! Please make sure there are no typo in those functions. Or you might want to rebuild service definitions if .proto file is changed."
            raise ModuleNotFoundError(message) from e
    return callables