import contextlib
from oslo_utils import uuidutils
from taskflow import logging
from taskflow.persistence import models
def temporary_log_book(backend=None):
    """Creates a temporary logbook for temporary usage in the given backend.

    Mainly useful for tests and other use cases where a temporary logbook
    is needed for a short-period of time.
    """
    book = models.LogBook('tmp')
    if backend is not None:
        with contextlib.closing(backend.get_connection()) as conn:
            conn.save_logbook(book)
    return book