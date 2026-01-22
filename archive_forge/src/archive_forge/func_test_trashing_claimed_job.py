import contextlib
import threading
from kazoo.protocol import paths as k_paths
from kazoo.recipe import watchers
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
import testtools
from zake import fake_client
from zake import utils as zake_utils
from taskflow import exceptions as excp
from taskflow.jobs.backends import impl_zookeeper
from taskflow import states
from taskflow import test
from taskflow.test import mock
from taskflow.tests.unit.jobs import base
from taskflow.tests import utils as test_utils
from taskflow.types import entity
from taskflow.utils import kazoo_utils
from taskflow.utils import misc
from taskflow.utils import persistence_utils as p_utils
def test_trashing_claimed_job(self):
    with base.connect_close(self.board):
        with self.flush(self.client):
            j = self.board.post('test', p_utils.temporary_log_book())
        self.assertEqual(states.UNCLAIMED, j.state)
        with self.flush(self.client):
            self.board.claim(j, self.board.name)
        self.assertEqual(states.CLAIMED, j.state)
        with self.flush(self.client):
            self.board.trash(j, self.board.name)
        trashed = []
        jobs = []
        paths = list(self.client.storage.paths.items())
        for path, value in paths:
            if path in self.bad_paths:
                continue
            if path.find(TRASH_FOLDER) > -1:
                trashed.append(path)
            elif path.find(self.board._job_base) > -1 and (not path.endswith(LOCK_POSTFIX)):
                jobs.append(path)
        self.assertEqual(1, len(trashed))
        self.assertEqual(0, len(jobs))