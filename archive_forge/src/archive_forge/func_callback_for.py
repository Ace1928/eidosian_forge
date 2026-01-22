from __future__ import annotations
from kombu.utils.eventio import ERR, READ, WRITE
from kombu.utils.functional import reprcall
def callback_for(h, fd, flag, *default):
    """Return the callback used for hub+fd+flag."""
    try:
        if flag & READ:
            return h.readers[fd]
        if flag & WRITE:
            if fd in h.consolidate:
                return h.consolidate_callback
            return h.writers[fd]
    except KeyError:
        if default:
            return default[0]
        raise