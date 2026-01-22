import ast
import collections
import contextlib
import functools
import itertools
import re
from numbers import Number
from typing import (
from more_itertools import windowed_complete
from typeguard import typechecked
from typing_extensions import Annotated, Literal
@typechecked
def plural_verb(self, text: Word, count: Optional[Union[str, int, Any]]=None) -> str:
    """
        Return the plural of text, where text is a verb.

        If count supplied, then return text if count is one of:
            1, a, an, one, each, every, this, that

        otherwise return the plural.

        Whitespace at the start and end is preserved.

        """
    pre, word, post = self.partition_word(text)
    if not word:
        return text
    plural = self.postprocess(word, self._pl_special_verb(word, count) or self._pl_general_verb(word, count))
    return f'{pre}{plural}{post}'