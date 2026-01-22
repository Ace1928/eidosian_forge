from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import hashlib
import os
import pkgutil
from unittest import mock
from gslib.exception import CommandException
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.hashing_helper import CalculateMd5FromContents
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.hashing_helper import HashingFileUploadWrapper
@mock.patch.object(hashlib, 'md5')
def testGetsMd5HashOnNonRedHatSystem(self, mock_md5):
    mock_md5.return_value = 'hash'
    self.assertEqual(GetMd5(b''), 'hash')
    mock_md5.assert_called_once_with(b'')