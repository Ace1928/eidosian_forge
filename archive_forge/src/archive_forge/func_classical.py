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
def classical(self, **kwargs) -> None:
    """
        turn classical mode on and off for various categories

        turn on all classical modes:
        classical()
        classical(all=True)

        turn on or off specific claassical modes:
        e.g.
        classical(herd=True)
        classical(names=False)

        By default all classical modes are off except names.

        unknown value in args or key in kwargs raises
        exception: UnknownClasicalModeError

        """
    if not kwargs:
        self.classical_dict = all_classical.copy()
        return
    if 'all' in kwargs:
        if kwargs['all']:
            self.classical_dict = all_classical.copy()
        else:
            self.classical_dict = no_classical.copy()
    for k, v in kwargs.items():
        if k in def_classical:
            self.classical_dict[k] = v
        else:
            raise UnknownClassicalModeError