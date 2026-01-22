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
def testValidSeekAway(self):
    for initial_read in (0, TRANSFER_BUFFER_SIZE - 1, TRANSFER_BUFFER_SIZE, TRANSFER_BUFFER_SIZE + 1, TRANSFER_BUFFER_SIZE * 2 - 1, TRANSFER_BUFFER_SIZE * 2, TRANSFER_BUFFER_SIZE * 2 + 1):
        self._testSeekAway(initial_read)