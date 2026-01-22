import threading
from .. import cethread, tests
def test_start_and_join_smoke_test(self):

    def do_nothing():
        pass
    tt = cethread.CatchingExceptionThread(target=do_nothing)
    tt.start()
    tt.join()