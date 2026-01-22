import contextlib
from oslo_utils import uuidutils
from taskflow import logging
from taskflow.persistence import models
def create_flow_detail(flow, book=None, backend=None, meta=None):
    """Creates a flow detail for a flow & adds & saves it in a logbook.

    This will create a flow detail for the given flow using the flow name,
    and add it to the provided logbook and then uses the given backend to save
    the logbook and then returns the created flow detail.

    If no book is provided a temporary one will be created automatically (no
    reference to the logbook will be returned, so this should nearly *always*
    be provided or only used in situations where no logbook is needed, for
    example in tests). If no backend is provided then no saving will occur and
    the created flow detail will not be persisted even if the flow detail was
    added to a given (or temporarily generated) logbook.
    """
    flow_id = uuidutils.generate_uuid()
    flow_name = getattr(flow, 'name', None)
    if flow_name is None:
        LOG.warning('No name provided for flow %s (id %s)', flow, flow_id)
        flow_name = flow_id
    flow_detail = models.FlowDetail(name=flow_name, uuid=flow_id)
    if meta is not None:
        if flow_detail.meta is None:
            flow_detail.meta = {}
        flow_detail.meta.update(meta)
    if backend is not None and book is None:
        LOG.warning('No logbook provided for flow %s, creating one.', flow)
        book = temporary_log_book(backend)
    if book is not None:
        book.add(flow_detail)
        if backend is not None:
            with contextlib.closing(backend.get_connection()) as conn:
                conn.save_logbook(book)
        return book.find(flow_id)
    else:
        return flow_detail