import contextlib
from oslo_utils import uuidutils
from taskflow import logging
from taskflow.persistence import models
def temporary_flow_detail(backend=None, meta=None):
    """Creates a temporary flow detail and logbook in the given backend.

    Mainly useful for tests and other use cases where a temporary flow detail
    and a temporary logbook is needed for a short-period of time.
    """
    flow_id = uuidutils.generate_uuid()
    book = temporary_log_book(backend)
    flow_detail = models.FlowDetail(name='tmp-flow-detail', uuid=flow_id)
    if meta is not None:
        if flow_detail.meta is None:
            flow_detail.meta = {}
        flow_detail.meta.update(meta)
    book.add(flow_detail)
    if backend is not None:
        with contextlib.closing(backend.get_connection()) as conn:
            conn.save_logbook(book)
    return (book, book.find(flow_id))