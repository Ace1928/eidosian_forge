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
def hundfn(self, hundreds: int, tens: int, units: int, mindex: int) -> str:
    if hundreds:
        andword = f' {self._number_args['andword']} ' if tens or units else ''
        return f'{unit[hundreds]} hundred{andword}{self.tenfn(tens, units)}{self.millfn(mindex)}, '
    if tens or units:
        return f'{self.tenfn(tens, units)}{self.millfn(mindex)}, '
    return ''