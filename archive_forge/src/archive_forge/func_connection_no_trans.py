from __future__ import annotations
import sqlalchemy as sa
from .. import assertions
from .. import config
from ..assertions import eq_
from ..util import drop_all_tables_from_metadata
from ... import Column
from ... import func
from ... import Integer
from ... import select
from ... import Table
from ...orm import DeclarativeBase
from ...orm import MappedAsDataclass
from ...orm import registry
@config.fixture()
def connection_no_trans(self):
    eng = getattr(self, 'bind', None) or config.db
    with eng.connect() as conn:
        yield conn