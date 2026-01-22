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
def download_range(self, destination_path, start_bytes, end_bytes=None, overwrite_existing=False, delete_on_failure=True):
    return self.driver.download_object_range(obj=self, destination_path=destination_path, start_bytes=start_bytes, end_bytes=end_bytes, overwrite_existing=overwrite_existing, delete_on_failure=delete_on_failure)