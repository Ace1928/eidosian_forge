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
def testSeekToBeginning(self):
    for num_bytes in (TRANSFER_BUFFER_SIZE - 1, TRANSFER_BUFFER_SIZE, TRANSFER_BUFFER_SIZE + 1, TRANSFER_BUFFER_SIZE * 2 - 1, TRANSFER_BUFFER_SIZE * 2, TRANSFER_BUFFER_SIZE * 2 + 1, TRANSFER_BUFFER_SIZE * 3 - 1, TRANSFER_BUFFER_SIZE * 3, TRANSFER_BUFFER_SIZE * 3 + 1):
        self._testSeekBack(num_bytes, num_bytes)