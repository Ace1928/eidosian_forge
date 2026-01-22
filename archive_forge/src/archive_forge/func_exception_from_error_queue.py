import os
import sys
import warnings
from typing import Any, Callable, NoReturn, Type, Union
from cryptography.hazmat.bindings.openssl.binding import Binding
def exception_from_error_queue(exception_type: Type[Exception]) -> NoReturn:
    """
    Convert an OpenSSL library failure into a Python exception.

    When a call to the native OpenSSL library fails, this is usually signalled
    by the return value, and an error code is stored in an error queue
    associated with the current thread. The err library provides functions to
    obtain these error codes and textual error messages.
    """
    errors = []
    while True:
        error = lib.ERR_get_error()
        if error == 0:
            break
        errors.append((text(lib.ERR_lib_error_string(error)), text(lib.ERR_func_error_string(error)), text(lib.ERR_reason_error_string(error))))
    raise exception_type(errors)