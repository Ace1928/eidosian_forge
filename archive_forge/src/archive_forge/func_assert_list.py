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
def assert_list(self, result, class_, list_):
    self.assert_(len(result) == len(list_), 'result list is not the same size as test list, ' + 'for class ' + class_.__name__)
    for i in range(0, len(list_)):
        self.assert_row(class_, result[i], list_[i])