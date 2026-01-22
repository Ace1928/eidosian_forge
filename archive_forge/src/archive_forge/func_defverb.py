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
def defverb(self, s1: Optional[Word], p1: Optional[Word], s2: Optional[Word], p2: Optional[Word], s3: Optional[Word], p3: Optional[Word]) -> int:
    """
        Set the verb plurals for s1, s2 and s3 to p1, p2 and p3 respectively.

        Where 1, 2 and 3 represent the 1st, 2nd and 3rd person forms of the verb.

        """
    self.checkpat(s1)
    self.checkpat(s2)
    self.checkpat(s3)
    self.checkpatplural(p1)
    self.checkpatplural(p2)
    self.checkpatplural(p3)
    self.pl_v_user_defined.extend((s1, p1, s2, p2, s3, p3))
    return 1