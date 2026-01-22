import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
def register_engine(engine):
    event.listen(engine, 'handle_error', handler, retval=True)

    @event.listens_for(engine, 'rollback_savepoint')
    def rollback_savepoint(conn, name, context):
        exc_info = sys.exc_info()
        if exc_info[1]:
            if not conn.invalidated:
                conn.info[ROLLBACK_CAUSE_KEY] = exc_info[1]
        del exc_info

    @event.listens_for(engine, 'rollback')
    @event.listens_for(engine, 'commit')
    def pop_exc_tx(conn):
        if not conn.invalidated:
            conn.info.pop(ROLLBACK_CAUSE_KEY, None)

    @event.listens_for(engine, 'checkin')
    def pop_exc_checkin(dbapi_conn, connection_record):
        connection_record.info.pop(ROLLBACK_CAUSE_KEY, None)