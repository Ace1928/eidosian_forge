from __future__ import annotations
from kombu.utils.eventio import ERR, READ, WRITE
from kombu.utils.functional import reprcall
def repr_readers(h):
    """Return description of pending readers."""
    return [f'({fd}){_rcb(cb)}->{repr_flag(READ | ERR)}' for fd, cb in h.readers.items()]