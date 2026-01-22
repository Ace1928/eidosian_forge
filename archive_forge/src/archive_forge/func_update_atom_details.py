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
def update_atom_details(self, atom_detail):
    try:
        atomdetails = self._tables.atomdetails
        with self._engine.begin() as conn:
            q = sql.select(atomdetails).where(atomdetails.c.uuid == atom_detail.uuid)
            row = conn.execute(q).first()
            if not row:
                raise exc.NotFound("No atom details found with uuid '%s'" % atom_detail.uuid)
            row = row._mapping
            e_ad = self._converter.convert_atom_detail(row)
            self._update_atom_details(conn, atom_detail, e_ad)
        return e_ad
    except sa_exc.SQLAlchemyError:
        exc.raise_with_cause(exc.StorageFailure, "Failed updating atom details with uuid '%s'" % atom_detail.uuid)