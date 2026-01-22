import abc
import contextlib
import os
import random
import tempfile
import testtools
import sqlalchemy as sa
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_sqlalchemy
from taskflow import test
from taskflow.tests.unit.persistence import base
def test_sqlite_persistence_entry_point(self):
    conf = {'connection': 'sqlite:///'}
    with contextlib.closing(backends.fetch(conf)) as be:
        self.assertIsInstance(be, impl_sqlalchemy.SQLAlchemyBackend)