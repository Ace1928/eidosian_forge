import copy
import functools
import threading
import time
from oslo_utils import strutils
import sqlalchemy as sa
from sqlalchemy import exc as sa_exc
from sqlalchemy import pool as sa_pool
from sqlalchemy import sql
import tenacity
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.persistence.backends.sqlalchemy import migration
from taskflow.persistence.backends.sqlalchemy import tables
from taskflow.persistence import base
from taskflow.persistence import models
from taskflow.utils import eventlet_utils
from taskflow.utils import misc
def destroy_logbook(self, book_uuid):
    try:
        logbooks = self._tables.logbooks
        with self._engine.begin() as conn:
            q = logbooks.delete().where(logbooks.c.uuid == book_uuid)
            r = conn.execute(q)
            if r.rowcount == 0:
                raise exc.NotFound("No logbook found with uuid '%s'" % book_uuid)
    except sa_exc.DBAPIError:
        exc.raise_with_cause(exc.StorageFailure, "Failed destroying logbook '%s'" % book_uuid)