from collections.abc import Generator
from contextlib import contextmanager
import pathlib
import tempfile
import pytest
from pandas.io.pytables import HDFStore
def safe_close(store):
    try:
        if store is not None:
            store.close()
    except OSError:
        pass