from __future__ import unicode_literals
from past.builtins import basestring
from .dag import KwargReprNode
from ._utils import escape_chars, get_hash_int
from builtins import object
import os
def get_stream_spec_nodes(stream_spec):
    stream_map = get_stream_map(stream_spec)
    return get_stream_map_nodes(stream_map)