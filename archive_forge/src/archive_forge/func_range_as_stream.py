import errno
import hashlib
import os.path  # pylint: disable-msg=W0404
import warnings
from typing import Dict, List, Type, Iterator, Optional
from os.path import join as pjoin
import libcloud.utils.files
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import BaseDriver, Connection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.storage.types import ObjectDoesNotExistError
def range_as_stream(self, start_bytes, end_bytes=None, chunk_size=None):
    return self.driver.download_object_range_as_stream(obj=self, start_bytes=start_bytes, end_bytes=end_bytes, chunk_size=chunk_size)