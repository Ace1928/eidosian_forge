from __future__ import annotations
from collections import defaultdict
import contextlib
from copy import copy
from itertools import filterfalse
import re
import sys
import warnings
from . import assertsql
from . import config
from . import engines
from . import mock
from .exclusions import db_spec
from .util import fail
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import types as sqltypes
from .. import util
from ..engine import default
from ..engine import url
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import decorator
def assert_unordered_result(self, result, cls, *expected):
    """As assert_result, but the order of objects is not considered.

        The algorithm is very expensive but not a big deal for the small
        numbers of rows that the test suite manipulates.
        """

    class immutabledict(dict):

        def __hash__(self):
            return id(self)
    found = util.IdentitySet(result)
    expected = {immutabledict(e) for e in expected}
    for wrong in filterfalse(lambda o: isinstance(o, cls), found):
        fail('Unexpected type "%s", expected "%s"' % (type(wrong).__name__, cls.__name__))
    if len(found) != len(expected):
        fail('Unexpected object count "%s", expected "%s"' % (len(found), len(expected)))
    NOVALUE = object()

    def _compare_item(obj, spec):
        for key, value in spec.items():
            if isinstance(value, tuple):
                try:
                    self.assert_unordered_result(getattr(obj, key), value[0], *value[1])
                except AssertionError:
                    return False
            elif getattr(obj, key, NOVALUE) != value:
                return False
        return True
    for expected_item in expected:
        for found_item in found:
            if _compare_item(found_item, expected_item):
                found.remove(found_item)
                break
        else:
            fail('Expected %s instance with attributes %s not found.' % (cls.__name__, repr(expected_item)))
    return True