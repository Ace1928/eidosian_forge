import io
import math
import os
import typing
import weakref
def get_page_labels(doc):
    """Return page label definitions in PDF document.

    Args:
        doc: PDF document (resp. 'self').
    Returns:
        A list of dictionaries with the following format:
        {'startpage': int, 'prefix': str, 'style': str, 'firstpagenum': int}.
    """
    return [rule_dict(item) for item in doc._get_page_labels()]