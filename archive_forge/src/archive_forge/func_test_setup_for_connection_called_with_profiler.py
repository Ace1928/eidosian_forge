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
def test_setup_for_connection_called_with_profiler(self):
    context_manager = enginefacade.transaction_context()
    context_manager.configure(connection='sqlite://')
    hook = mock.Mock()
    context_manager.append_on_engine_create(hook)
    self.assertEqual([hook], context_manager._factory._facade_cfg['on_engine_create'])

    @context_manager.reader
    def go(context):
        hook.assert_called_once_with(context.session.bind)
    go(oslo_context.RequestContext())