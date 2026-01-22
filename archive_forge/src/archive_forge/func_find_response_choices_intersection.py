import functools
from dataclasses import asdict, dataclass, field, replace
from itertools import zip_longest
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from outlines.base import vectorize
from outlines.caching import cache
def find_response_choices_intersection(response: List[int], choices: List[List[int]]) -> Tuple[List[int], List[List[int]]]:
    """Find the longest intersection between the response and the different
    choices.

    Say the response is of the form `[1, 2, 3, 4, 5]` and we have the choices
    `[[1, 2], [1, 2, 3], [6, 7, 8]` then the function will return `[1, 2, 3]` as the
    intersection, and `[[]]` as the list of choices left.

    Parameters
    ----------
    response
        The model's response
    choices
        The remaining possible choices

    Returns
    -------
    A tuple that contains the longest intersection between the response and the
    different choices, and the choices which start with this intersection, with the
    intersection removed.

    """
    max_len_prefix = 0
    choices_left = []
    longest_prefix = []
    for i, choice in enumerate(choices):
        prefix = find_longest_intersection(response, choice)
        if len(prefix) > max_len_prefix:
            max_len_prefix = len(prefix)
            choices_left = [choice[len(prefix):]]
            longest_prefix = prefix
        elif len(prefix) == max_len_prefix:
            choices_left.append(choice[len(prefix):])
    return (longest_prefix, choices_left)