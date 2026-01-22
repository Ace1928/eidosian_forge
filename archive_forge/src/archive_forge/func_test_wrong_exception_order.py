import os
import unittest
import unittest.mock as mock
from urllib.error import HTTPError
from distutils.command import upload as upload_mod
from distutils.command.upload import upload
from distutils.core import Distribution
from distutils.errors import DistutilsError
from distutils.log import ERROR, INFO
from distutils.tests.test_config import PYPIRC, BasePyPIRCCommandTestCase
def test_wrong_exception_order(self):
    tmp = self.mkdtemp()
    path = os.path.join(tmp, 'xxx')
    self.write_file(path)
    dist_files = [('xxx', '2.6', path)]
    self.write_file(self.rc, PYPIRC_LONG_PASSWORD)
    pkg_dir, dist = self.create_dist(dist_files=dist_files)
    tests = [(OSError('oserror'), 'oserror', OSError), (HTTPError('url', 400, 'httperror', {}, None), 'Upload failed (400): httperror', DistutilsError)]
    for exception, expected, raised_exception in tests:
        with self.subTest(exception=type(exception).__name__):
            with mock.patch('distutils.command.upload.urlopen', new=mock.Mock(side_effect=exception)):
                with self.assertRaises(raised_exception):
                    cmd = upload(dist)
                    cmd.ensure_finalized()
                    cmd.run()
                results = self.get_logs(ERROR)
                self.assertIn(expected, results[-1])
                self.clear_logs()