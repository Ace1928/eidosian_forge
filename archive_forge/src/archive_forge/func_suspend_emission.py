import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config
@contextmanager
def suspend_emission(builder):
    """Suspends the emission of debug_metadata for the duration of the context
    managed block."""
    ref = builder.debug_metadata
    builder.debug_metadata = None
    try:
        yield
    finally:
        builder.debug_metadata = ref