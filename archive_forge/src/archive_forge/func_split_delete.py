import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def split_delete(chunks):
    """ Returns (stuff_before_DEL_START, stuff_inside_DEL_START_END,
    stuff_after_DEL_END).  Returns the first case found (there may be
    more DEL_STARTs in stuff_after_DEL_END).  Raises NoDeletes if
    there's no DEL_START found. """
    try:
        pos = chunks.index(DEL_START)
    except ValueError:
        raise NoDeletes
    pos2 = chunks.index(DEL_END)
    return (chunks[:pos], chunks[pos + 1:pos2], chunks[pos2 + 1:])