import contextlib
import threading
import time
from taskflow import exceptions as excp
from taskflow.persistence.backends import impl_dir
from taskflow import states
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import threading_utils
def test_fresh_iter(self):
    with connect_close(self.board):
        book = p_utils.temporary_log_book()
        self.board.post('test', book)
        jobs = list(self.board.iterjobs(ensure_fresh=True))
        self.assertEqual(1, len(jobs))