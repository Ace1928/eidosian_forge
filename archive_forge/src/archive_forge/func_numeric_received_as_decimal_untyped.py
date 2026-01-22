from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def numeric_received_as_decimal_untyped(self):
    """target backend will return result columns that are explicitly
        against NUMERIC or similar precision-numeric datatypes (not including
        FLOAT or INT types) as Python Decimal objects, and not as floats
        or ints, including when no SQLAlchemy-side typing information is
        associated with the statement (e.g. such as a raw SQL string).

        This should be enabled if either the DBAPI itself returns Decimal
        objects, or if the dialect has set up DBAPI-specific return type
        handlers such that Decimal objects come back automatically.

        """
    return exclusions.open()