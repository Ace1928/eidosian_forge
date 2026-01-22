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
@mock.patch.object(Session, 'commit')
@mock.patch.object(Session, 'rollback')
def test_save_and_reraise_when_rollback_exception(self, rollback_patch, commit_patch):
    context = oslo_context.RequestContext()
    log = self.useFixture(fixtures.FakeLogger())

    class RollbackException(Exception):
        pass

    class CommitException(Exception):
        pass
    commit_patch.side_effect = CommitException()
    rollback_patch.side_effect = RollbackException()

    @enginefacade.writer
    def go_session(context):
        context.session.add(self.User(name='u1'))
    self.assertRaises(RollbackException, go_session, context)
    self.assertIn('CommitException', log.output)