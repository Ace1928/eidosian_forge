import collections.abc
import gc
import inspect
import re
import sys
import weakref
from functools import partial, wraps
from itertools import chain
from typing import (
from scrapy.utils.asyncgen import as_async_generator
def re_rsearch(pattern: Union[str, Pattern], text: str, chunk_size: int=1024) -> Optional[Tuple[int, int]]:
    """
    This function does a reverse search in a text using a regular expression
    given in the attribute 'pattern'.
    Since the re module does not provide this functionality, we have to find for
    the expression into chunks of text extracted from the end (for the sake of efficiency).
    At first, a chunk of 'chunk_size' kilobytes is extracted from the end, and searched for
    the pattern. If the pattern is not found, another chunk is extracted, and another
    search is performed.
    This process continues until a match is found, or until the whole file is read.
    In case the pattern wasn't found, None is returned, otherwise it returns a tuple containing
    the start position of the match, and the ending (regarding the entire text).
    """

    def _chunk_iter() -> Generator[Tuple[str, int], Any, None]:
        offset = len(text)
        while True:
            offset -= chunk_size * 1024
            if offset <= 0:
                break
            yield (text[offset:], offset)
        yield (text, 0)
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    for chunk, offset in _chunk_iter():
        matches = [match for match in pattern.finditer(chunk)]
        if matches:
            start, end = matches[-1].span()
            return (offset + start, offset + end)
    return None