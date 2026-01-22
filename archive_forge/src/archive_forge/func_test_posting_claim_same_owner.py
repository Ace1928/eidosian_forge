import time
from unittest import mock
from oslo_utils import uuidutils
import testtools
from taskflow import exceptions as excp
from taskflow.jobs.backends import impl_redis
from taskflow import states
from taskflow import test
from taskflow.tests.unit.jobs import base
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import redis_utils as ru
def test_posting_claim_same_owner(self):
    with base.connect_close(self.board):
        with self.flush(self.client):
            self.board.post('test', p_utils.temporary_log_book())
        self.assertEqual(1, self.board.job_count)
        possible_jobs = list(self.board.iterjobs(only_unclaimed=True))
        self.assertEqual(1, len(possible_jobs))
        j = possible_jobs[0]
        self.assertEqual(states.UNCLAIMED, j.state)
        with self.flush(self.client):
            self.board.claim(j, self.board.name)
        possible_jobs = list(self.board.iterjobs())
        self.assertEqual(1, len(possible_jobs))
        with self.flush(self.client):
            self.assertRaises(excp.UnclaimableJob, self.board.claim, possible_jobs[0], self.board.name)
        possible_jobs = list(self.board.iterjobs(only_unclaimed=True))
        self.assertEqual(0, len(possible_jobs))