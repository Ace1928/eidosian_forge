from __future__ import annotations
import contextlib
import re
import sys
from typing import Any
from typing import Dict
from sqlalchemy import exc as sa_exc
from sqlalchemy.engine import default
from sqlalchemy.testing.assertions import _expect_warnings
from sqlalchemy.testing.assertions import eq_  # noqa
from sqlalchemy.testing.assertions import is_  # noqa
from sqlalchemy.testing.assertions import is_false  # noqa
from sqlalchemy.testing.assertions import is_not_  # noqa
from sqlalchemy.testing.assertions import is_true  # noqa
from sqlalchemy.testing.assertions import ne_  # noqa
from sqlalchemy.util import decorator
from ..util import sqla_compat
def expect_sqlalchemy_deprecated(*messages, **kw):
    return _expect_warnings(sa_exc.SADeprecationWarning, messages, **kw)