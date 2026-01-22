import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def merge_delete(del_chunks, doc):
    """ Adds the text chunks in del_chunks to the document doc (another
    list of text chunks) with marker to show it is a delete.
    cleanup_delete later resolves these markers into <del> tags."""
    doc.append(DEL_START)
    doc.extend(del_chunks)
    doc.append(DEL_END)