from __future__ import annotations
from argparse import Namespace
import collections
import inspect
import typing
from typing import Any
from typing import Callable
from typing import Iterable
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from . import mock
from . import requirements as _requirements
from .util import fail
from .. import util
@classmethod
def set_as_current(cls, config, namespace):
    global db, _current, db_url, test_schema, test_schema_2, db_opts
    _current = config
    db_url = config.db.url
    db_opts = config.db_opts
    test_schema = config.test_schema
    test_schema_2 = config.test_schema_2
    namespace.db = db = config.db