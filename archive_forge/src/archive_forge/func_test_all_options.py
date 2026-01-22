import collections
import contextlib
import copy
import fixtures
import pickle
import sys
from unittest import mock
import warnings
from oslo_config import cfg
from oslo_context import context as oslo_context
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import engines as oslo_engines
from oslo_db.sqlalchemy import orm
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
from oslo_db import warning
def test_all_options(self):
    """test that everything in CONF.database.iteritems() is accepted.

        There's a handful of options in oslo.db.options that seem to have
        no meaning, but need to be accepted.   In particular, Cinder and
        maybe others are doing exactly this call.

        """
    factory = enginefacade._TransactionFactory()
    cfg.CONF.register_opts(options.database_opts, 'database')
    factory.configure(**dict(cfg.CONF.database.items()))