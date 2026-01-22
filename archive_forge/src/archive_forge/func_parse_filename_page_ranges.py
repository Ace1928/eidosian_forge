import re
from typing import Any, List, Tuple, Union
from .errors import ParseError
def parse_filename_page_ranges(args: List[Union[str, PageRange, None]]) -> List[Tuple[str, PageRange]]:
    """
    Given a list of filenames and page ranges, return a list of (filename, page_range) pairs.

    Args:
        args: A list where the first element is a filename. The other elements are
            filenames, page-range expressions, slice objects, or PageRange objects.
            A filename not followed by a page range indicates all pages of the file.

    Returns:
        A list of (filename, page_range) pairs.
    """
    pairs: List[Tuple[str, PageRange]] = []
    pdf_filename = None
    did_page_range = False
    for arg in args + [None]:
        if PageRange.valid(arg):
            if not pdf_filename:
                raise ValueError('The first argument must be a filename, not a page range.')
            pairs.append((pdf_filename, PageRange(arg)))
            did_page_range = True
        else:
            if pdf_filename and (not did_page_range):
                pairs.append((pdf_filename, PAGE_RANGE_ALL))
            pdf_filename = arg
            did_page_range = False
    return pairs