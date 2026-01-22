import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, Union
from ._version import get_versions
def set_execution(engine: Any=None, storage_format: Any=None) -> Tuple['Engine', 'StorageFormat']:
    """
    Method to set the _pair_ of execution engine and storage format format simultaneously.
    This is needed because there might be cases where switching one by one would be
    impossible, as not all pairs of values are meaningful.

    The method returns pair of old values, so it is easy to return back.
    """
    from .config import Engine, StorageFormat
    old_engine, old_storage_format = (None, None)
    if engine is not None:
        old_engine = Engine._put_nocallback(engine)
    if storage_format is not None:
        old_storage_format = StorageFormat._put_nocallback(storage_format)
    if old_engine is not None:
        Engine._check_callbacks(old_engine)
    if old_storage_format is not None:
        StorageFormat._check_callbacks(old_storage_format)
    return (old_engine, old_storage_format)