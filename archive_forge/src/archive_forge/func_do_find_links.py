import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def do_find_links(text: str) -> List[Tuple[int, int]]:
    """Determine if docstring contains any links.

    Parameters
    ----------
    text : str
        The docstring description to check for link patterns.

    Returns
    -------
    url_index : list
        A list of tuples with each tuple containing the starting and ending
        position of each URL found in the passed description.
    """
    _url_iter = re.finditer(URL_REGEX, text)
    return [(_url.start(0), _url.end(0)) for _url in _url_iter]