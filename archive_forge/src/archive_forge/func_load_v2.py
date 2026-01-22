import os
import re
import zlib
from typing import IO, TYPE_CHECKING, Callable, Iterator
from sphinx.util import logging
from sphinx.util.typing import Inventory
@classmethod
def load_v2(cls, stream: InventoryFileReader, uri: str, join: Callable) -> Inventory:
    invdata: Inventory = {}
    projname = stream.readline().rstrip()[11:]
    version = stream.readline().rstrip()[11:]
    line = stream.readline()
    if 'zlib' not in line:
        raise ValueError('invalid inventory header (not compressed): %s' % line)
    for line in stream.read_compressed_lines():
        m = re.match('(?x)(.+?)\\s+(\\S+)\\s+(-?\\d+)\\s+?(\\S*)\\s+(.*)', line.rstrip())
        if not m:
            continue
        name, type, prio, location, dispname = m.groups()
        if ':' not in type:
            continue
        if type == 'py:module' and type in invdata and (name in invdata[type]):
            continue
        if location.endswith('$'):
            location = location[:-1] + name
        location = join(uri, location)
        invdata.setdefault(type, {})[name] = (projname, version, location, dispname)
    return invdata