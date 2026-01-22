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
def present_participle(self, word: Word) -> str:
    """
        Return the present participle for word.

        word is the 3rd person singular verb.

        """
    plv = self.plural_verb(word, 2)
    ans = plv
    for regexen, repl in PRESENT_PARTICIPLE_REPLACEMENTS:
        ans, num = regexen.subn(repl, plv)
        if num:
            return f'{ans}ing'
    return f'{ans}ing'