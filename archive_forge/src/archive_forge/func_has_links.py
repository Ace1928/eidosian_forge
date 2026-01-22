import io
import math
import os
import typing
import weakref
def has_links(doc: fitz.Document) -> bool:
    """Check whether there are links on any page."""
    if doc.is_closed:
        raise ValueError('document closed')
    if not doc.is_pdf:
        raise ValueError('is no PDF')
    for i in range(doc.page_count):
        for item in doc.page_annot_xrefs(i):
            if item[1] == fitz.PDF_ANNOT_LINK:
                return True
    return False