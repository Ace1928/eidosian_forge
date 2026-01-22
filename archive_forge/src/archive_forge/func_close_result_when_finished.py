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
def close_result_when_finished(self):
    to_close = []
    to_consume = []

    def go(result, consume=False):
        to_close.append(result)
        if consume:
            to_consume.append(result)
    yield go
    for r in to_consume:
        try:
            r.all()
        except:
            pass
    for r in to_close:
        try:
            r.close()
        except:
            pass