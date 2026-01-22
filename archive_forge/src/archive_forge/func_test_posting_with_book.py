import contextlib
import threading
import time
from taskflow import exceptions as excp
from taskflow.persistence.backends import impl_dir
from taskflow import states
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import threading_utils
def test_posting_with_book(self):
    backend = impl_dir.DirBackend(conf={'path': self.makeTmpDir()})
    backend.get_connection().upgrade()
    book, flow_detail = p_utils.temporary_flow_detail(backend)
    self.assertEqual(1, len(book))
    client, board = self.create_board(persistence=backend)
    with connect_close(board):
        with self.flush(client):
            board.post('test', book)
        possible_jobs = list(board.iterjobs(only_unclaimed=True))
        self.assertEqual(1, len(possible_jobs))
        j = possible_jobs[0]
        self.assertEqual(1, len(j.book))
        self.assertEqual(book.name, j.book.name)
        self.assertEqual(book.uuid, j.book.uuid)
        self.assertEqual(book.name, j.book_name)
        self.assertEqual(book.uuid, j.book_uuid)
        flow_details = list(j.book)
        self.assertEqual(flow_detail.uuid, flow_details[0].uuid)
        self.assertEqual(flow_detail.name, flow_details[0].name)