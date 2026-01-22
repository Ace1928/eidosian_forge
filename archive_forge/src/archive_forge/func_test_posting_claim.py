import contextlib
import threading
import time
from taskflow import exceptions as excp
from taskflow.persistence.backends import impl_dir
from taskflow import states
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import threading_utils
def test_posting_claim(self):
    with connect_close(self.board):
        with self.flush(self.client):
            self.board.post('test', p_utils.temporary_log_book())
        self.assertEqual(1, self.board.job_count)
        possible_jobs = list(self.board.iterjobs(only_unclaimed=True))
        self.assertEqual(1, len(possible_jobs))
        j = possible_jobs[0]
        self.assertEqual(states.UNCLAIMED, j.state)
        with self.flush(self.client):
            self.board.claim(j, self.board.name)
        self.assertEqual(self.board.name, self.board.find_owner(j))
        self.assertEqual(states.CLAIMED, j.state)
        possible_jobs = list(self.board.iterjobs(only_unclaimed=True))
        self.assertEqual(0, len(possible_jobs))
    self.close_client(self.client)
    self.assertRaisesAttrAccess(excp.JobFailure, j, 'state')