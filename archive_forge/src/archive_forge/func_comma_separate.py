from __future__ import annotations
import keyword
import warnings
from typing import Collection, List, Mapping, Optional, Set, Tuple, Union
def comma_separate(collection: Collection[Union[str, Collection[str]]]) -> str:
    """Convert a collection of strings representing first class dims into a comma-separated string.

    Args:
        collection (Collection[Union[str, Collection[str]]]): the collection of strings to convert

    Returns:
        str: the comma-separated string

    Examples:
        >>> comma_separate(('d0',))
        'd0'

        >>> comma_separate(('d0', 'd1', 'd2', 'd3'))
        'd0, d1, d2, d3'

        >>> comma_separate([('d1', 'd4')])
        '(d1, d4)'

        >>> comma_separate([('d0',), (), ('d1',), ('d2',), ('d3', 'd4')])
        '(d0,), (), (d1,), (d2,), (d3, d4)'
    """
    return ', '.join((item if isinstance(item, str) else f'({comma_separate(item)}{(',' if len(item) == 1 else '')})' for item in collection))