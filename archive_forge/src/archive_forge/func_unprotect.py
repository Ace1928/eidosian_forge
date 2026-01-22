import contextlib
import typing
from . import openrlib
from _cffi_backend import FFI  # type: ignore
def unprotect(self, n: int) -> None:
    """Release the n objects last added to the protection stack."""
    if n > self._counter:
        raise ValueError('n > count')
    self._counter -= n
    openrlib.rlib.Rf_unprotect(n)