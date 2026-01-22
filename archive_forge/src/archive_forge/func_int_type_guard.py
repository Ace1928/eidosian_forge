import sys
from contextlib import contextmanager
from typing import (
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def int_type_guard(self, x) -> TypeGuard[int]:
    return isinstance(x, int)