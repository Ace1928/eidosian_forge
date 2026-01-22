import io
import sys
import requests
import testtools
from glanceclient.common import progressbar
from glanceclient.common import utils
from glanceclient.tests import utils as test_utils
def test_iter_file_display_progress_bar(self):
    size = 98304
    file_obj = io.StringIO('X' * size)
    saved_stdout = sys.stdout
    try:
        sys.stdout = output = test_utils.FakeTTYStdout()
        file_obj = progressbar.VerboseFileWrapper(file_obj, size)
        chunksize = 1024
        chunk = file_obj.read(chunksize)
        while chunk:
            chunk = file_obj.read(chunksize)
        self.assertEqual('[%s>] 100%%\n' % ('=' * 29), output.getvalue())
    finally:
        sys.stdout = saved_stdout