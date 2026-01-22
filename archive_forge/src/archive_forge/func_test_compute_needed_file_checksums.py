from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
from gslib.commands.rsync import _ComputeNeededFileChecksums
from gslib.commands.rsync import _NA
from gslib.tests.testcase.unit_testcase import GsUtilUnitTestCase
from gslib.utils.hashing_helper import CalculateB64EncodedCrc32cFromContents
from gslib.utils.hashing_helper import CalculateB64EncodedMd5FromContents
def test_compute_needed_file_checksums(self):
    """Tests that we compute all/only needed file checksums."""
    size = 4
    logger = logging.getLogger()
    tmpdir = self.CreateTempDir()
    file_url_str = 'file://%s' % os.path.join(tmpdir, 'obj1')
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1')
    cloud_url_str = 'gs://whatever'
    with open(os.path.join(tmpdir, 'obj1'), 'rb') as fp:
        crc32c = CalculateB64EncodedCrc32cFromContents(fp)
        fp.seek(0)
        md5 = CalculateB64EncodedMd5FromContents(fp)
    src_crc32c, src_md5, dst_crc32c, dst_md5 = _ComputeNeededFileChecksums(logger, file_url_str, size, _NA, _NA, cloud_url_str, size, crc32c, _NA)
    self.assertEqual(crc32c, src_crc32c)
    self.assertEqual(_NA, src_md5)
    self.assertEqual(crc32c, dst_crc32c)
    self.assertEqual(_NA, dst_md5)
    src_crc32c, src_md5, dst_crc32c, dst_md5 = _ComputeNeededFileChecksums(logger, file_url_str, size, _NA, _NA, cloud_url_str, size, _NA, md5)
    self.assertEqual(_NA, src_crc32c)
    self.assertEqual(md5, src_md5)
    self.assertEqual(_NA, dst_crc32c)
    self.assertEqual(md5, dst_md5)
    src_crc32c, src_md5, dst_crc32c, dst_md5 = _ComputeNeededFileChecksums(logger, cloud_url_str, size, crc32c, _NA, file_url_str, size, _NA, _NA)
    self.assertEqual(crc32c, dst_crc32c)
    self.assertEqual(_NA, src_md5)
    self.assertEqual(crc32c, src_crc32c)
    self.assertEqual(_NA, src_md5)
    src_crc32c, src_md5, dst_crc32c, dst_md5 = _ComputeNeededFileChecksums(logger, cloud_url_str, size, _NA, md5, file_url_str, size, _NA, _NA)
    self.assertEqual(_NA, dst_crc32c)
    self.assertEqual(md5, src_md5)
    self.assertEqual(_NA, src_crc32c)
    self.assertEqual(md5, src_md5)