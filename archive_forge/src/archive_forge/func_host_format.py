from __future__ import annotations
import os
import socket
from functools import partial
from kombu.entity import Exchange, Queue
from .functional import memoize
from .text import simple_format
def host_format(s: str, host: str | None=None, name: str | None=None, **extra: dict) -> str:
    """Format host %x abbreviations."""
    host = host or gethostname()
    hname, _, domain = host.partition('.')
    name = name or hname
    keys = dict({'h': host, 'n': name, 'd': domain, 'i': _fmt_process_index, 'I': _fmt_process_index_with_prefix}, **extra)
    return simple_format(s, keys)