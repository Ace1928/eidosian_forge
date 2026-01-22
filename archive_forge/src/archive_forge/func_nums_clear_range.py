from typing import Iterator, List, Optional, Tuple, cast
from ._protocols import PdfCommonDocProtocol
from ._utils import logger_warning
from .generic import ArrayObject, DictionaryObject, NullObject, NumberObject
def nums_clear_range(key: NumberObject, page_index_to: int, nums: ArrayObject) -> None:
    """
    Remove all entries in a number tree in a range after an entry.

    See 7.9.7 "Number Trees".

    Args:
        key: number key of the entry before the range
        page_index_to: The page index of the upper limit of the range
        nums: Nums array to modify
    """
    if len(nums) % 2 != 0:
        raise ValueError('a nums like array must have an even number of elements')
    if page_index_to < key:
        raise ValueError('page_index_to must be greater or equal than key')
    i = nums.index(key) + 2
    while i < len(nums) and nums[i] <= page_index_to:
        nums.pop(i)
        nums.pop(i)