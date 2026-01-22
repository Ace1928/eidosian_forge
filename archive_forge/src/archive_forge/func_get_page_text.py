import io
import math
import os
import typing
import weakref
def get_page_text(doc: fitz.Document, pno: int, option: str='text', clip: rect_like=None, flags: OptInt=None, textpage: fitz.TextPage=None, sort: bool=False) -> typing.Any:
    """Extract a document page's text by page number.

    Notes:
        Convenience function calling page.get_text().
    Args:
        pno: page number
        option: (str) text, words, blocks, html, dict, json, rawdict, xhtml or xml.
    Returns:
        output from page.TextPage().
    """
    return doc[pno].get_text(option, clip=clip, flags=flags, sort=sort)