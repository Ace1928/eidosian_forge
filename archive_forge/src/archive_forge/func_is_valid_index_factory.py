import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.rusty import Err, Ok, Result
from black.strings import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def is_valid_index_factory(seq: Sequence[Any]) -> Callable[[int], bool]:
    """
    Examples:
        ```
        my_list = [1, 2, 3]

        is_valid_index = is_valid_index_factory(my_list)

        assert is_valid_index(0)
        assert is_valid_index(2)

        assert not is_valid_index(3)
        assert not is_valid_index(-1)
        ```
    """

    def is_valid_index(idx: int) -> bool:
        """
        Returns:
            True iff @idx is positive AND seq[@idx] does NOT raise an
            IndexError.
        """
        return 0 <= idx < len(seq)
    return is_valid_index