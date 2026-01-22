import contextlib
import itertools
from unittest import mock
import sqlalchemy as sqla
from sqlalchemy import event
import sqlalchemy.exc
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy import sql
from oslo_db import exception
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import exc_filters
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
from oslo_db.tests import utils as test_utils
Fixture for testing the ping listener.

        For SQLAlchemy 2.0, the mocking is placed more deeply in the
        stack within the DBAPI connection / cursor so that we can also
        effectively mock out the "pre ping" condition.

        :param dialect_name: dialect to use.  "postgresql" or "mysql"
        :param exception: an exception class to raise
        :param db_stays_down: if True, the database will stay down after the
         first ping fails
        :param is_disconnect: whether or not the SQLAlchemy dialect should
         consider the exception object as a "disconnect error".   Openstack's
         own exception handlers upgrade various DB exceptions to be
         "disconnect" scenarios that SQLAlchemy itself does not, such as
         some specific Galera error messages.

        The importance of an exception being a "disconnect error" means that
        SQLAlchemy knows it can discard the connection and then reconnect.
        If the error is not a "disconnection error", then it raises.
        