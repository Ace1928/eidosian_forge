from __future__ import print_function
import email.utils
import errno
import hashlib
import mimetypes
import os
import re
import base64
import binascii
import math
from hashlib import md5
import boto.utils
from boto.compat import BytesIO, six, urllib, encodebytes
from boto.exception import BotoClientError
from boto.exception import StorageDataError
from boto.exception import PleaseRetryException
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.provider import Provider
from boto.s3.keyfile import KeyFile
from boto.s3.user import User
from boto import UserAgent
import boto.utils
from boto.utils import compute_md5, compute_hash
from boto.utils import find_matching_headers
from boto.utils import merge_headers_by_name
from boto.utils import print_to_fd
def compute_md5(self, fp, size=None):
    """
        :type fp: file
        :param fp: File pointer to the file to MD5 hash.  The file
            pointer will be reset to the same position before the
            method returns.

        :type size: int
        :param size: (optional) The Maximum number of bytes to read
            from the file pointer (fp). This is useful when uploading
            a file in multiple parts where the file is being split
            in place into different parts. Less bytes may be available.
        """
    hex_digest, b64_digest, data_size = compute_md5(fp, size=size)
    self.size = data_size
    return (hex_digest, b64_digest)